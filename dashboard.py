import math
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st
from engine import LEAGUES, fetch_season, odds_triplet, implied_probabilities, backtest

def season_code(y):
    return f'{y%100:02d}{(y+1)%100:02d}'

def evaluate(df,start,end):
    hist=defaultdict(list); rows=[]
    for date, day in df.sort_values('Date').groupby('Date'):
        for _,r in day.iterrows():
            if not start<=date<=end: continue
            h=hist[(r.LeagueCode,r.HomeTeam,'H')][-10:]; a=hist[(r.LeagueCode,r.AwayTeam,'A')][-10:]
            if len(h)<5 or len(a)<5: continue
            op,_=odds_triplet(r); cl,_=odds_triplet(r,True)
            if op is None: continue
            market=implied_probabilities(op)
            if market is None: continue
            # Shrink venue-specific scoring/conceding rates toward fixed priors.
            def rate(xs,i,prior): return (sum(x[i] for x in xs)+5*prior)/(len(xs)+5)
            lh=(rate(h,0,1.5)+rate(a,1,1.5))/2; la=(rate(a,0,1.2)+rate(h,1,1.2))/2
            def poisson(l): return np.array([math.exp(-l)*l**k/math.factorial(k) for k in range(16)])
            mat=np.outer(poisson(lh),poisson(la)); mat/=mat.sum()
            p=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()]); blend=.5*p+.5*market
            actual=0 if r.FTHG>r.FTAG else 2 if r.FTHG<r.FTAG else 1
            score=np.unravel_index(mat.argmax(),mat.shape)
            rr=dict(date=date.date(),league=r.League,home=r.HomeTeam,away=r.AwayTeam,actual_score=f'{int(r.FTHG)}-{int(r.FTAG)}',modal_score=f'{score[0]}-{score[1]}',home_goals=lh,away_goals=la,btts=mat[1:,1:].sum(),over_15=sum(mat[i,j] for i in range(16) for j in range(16) if i+j>=2),over_25=sum(mat[i,j] for i in range(16) for j in range(16) if i+j>=3),actual_btts=int(r.FTHG>0 and r.FTAG>0),actual_over_25=int(r.FTHG+r.FTAG>=3))
            for name,prob in [('market',market),('goals',p),('blend',blend)]:
                pick=int(prob.argmax()); rr.update({f'{name}_pick':['Home','Draw','Away'][pick],f'{name}_correct':int(pick==actual),f'{name}_brier':float(((prob-np.eye(3)[actual])**2).sum()),f'{name}_logloss':-math.log(max(prob[actual],1e-12)),f'{name}_return':op[pick]-1 if pick==actual else -1})
                for k,label in enumerate(['home','draw','away']): rr[f'{name}_{label}']=prob[k]
            rr['research_signal']='No reliable signal' # no validated decision threshold yet
            rows.append(rr)
        # Update only after all matches on this date have been predicted.
        for _,r in day.iterrows():
            hist[(r.LeagueCode,r.HomeTeam,'H')].append((r.FTHG,r.FTAG)); hist[(r.LeagueCode,r.AwayTeam,'A')].append((r.FTAG,r.FTHG))
    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title='Football Intelligence V3',layout='wide')
    if st.sidebar.radio('Workspace', ['Historical backtest','Market research']) == 'Market research':
        market_research()
        return
    st.title('Football Intelligence V3 — Historical backtest')
    st.caption('Historical research • 10 European leagues • No live trading or verified profitable signals')
    with st.sidebar:
        start=pd.Timestamp(st.date_input('Test start',pd.Timestamp('2026-08-24'))); end=pd.Timestamp(st.date_input('Test end',pd.Timestamp('2026-11-22')))
        leagues=st.multiselect('Leagues',list(LEAGUES),default=list(LEAGUES),format_func=lambda x:LEAGUES[x]); go=st.button('Load and analyze',type='primary')
    if go:
        if start>end or not leagues: st.error('Choose valid dates and leagues.'); return
        frames=[]; errors=[]
        first=start.year-(start.month<7)-1; last=end.year-(end.month<7)
        with st.spinner('Loading seasons and running chronological evaluation…'):
            for y in range(first,last+1):
                try:
                    # Refresh current-season cache each run.
                    from pathlib import Path
                    if y==last:
                        for l in leagues: Path(f'data/{season_code(y)}_{l}.csv').unlink(missing_ok=True)
                    d,e=fetch_season(season_code(y),leagues); frames.append(d); errors.extend(e)
                except Exception as exc: errors.append(str(exc))
            if not frames: st.error('No data available'); return
            df=pd.concat(frames).drop_duplicates(['Date','LeagueCode','HomeTeam','AwayTeam']); result=evaluate(df,start,end); old,summary=backtest(df,str(start.date()),str(end.date()))
        st.session_state['v2']=(result,summary,old,errors,int(df.Date.between(start,end).sum()))
    if 'v2' not in st.session_state: st.info('Open sidebar and select Load and analyze.'); return
    r,s,old,errors,total=st.session_state['v2']; st.write(f'Eligible matches: {len(r)} / {total} completed matches loaded')
    if errors: st.warning('Data gaps: '+'; '.join(errors))
    if r.empty: st.warning('No eligible matches with sufficient prior form and opening odds.'); return
    st.subheader('Same-match model comparison')
    stats=[]
    for n in ['market','goals','blend']:
        stats.append({'Model':n,'Accuracy':r[n+'_correct'].mean(),'Brier (lower better)':r[n+'_brier'].mean(),'Log loss (lower better)':r[n+'_logloss'].mean(),'Hypothetical ROI':r[n+'_return'].mean()})
    st.dataframe(pd.DataFrame(stats),hide_index=True); st.caption('Fixed one-unit hypothetical picks at recorded opening odds; execution not verified. No tuning on test results. Lower Brier/log loss is better.')
    st.write('Goals model: BTTS Brier',float(((r.btts-r.actual_btts)**2).mean()),'Over 2.5 Brier',float(((r.over_25-r.actual_over_25)**2).mean()))
    st.subheader('Match probabilities'); st.dataframe(r,hide_index=True); st.download_button('Download V2 results',r.to_csv(index=False),'v2_results.csv','text/csv')
    with st.expander('Original odds movement diagnostic (different sample)'):
        st.json(s); st.dataframe(old)
    st.warning('Research only. No validated signal threshold. Scores are modal estimates, not guarantees. No xG, injuries, exchange trades or live odds integrated yet.')


def market_research():
    import requests, json
    from datetime import datetime, timezone
    st.title('Market research — V3 preview')
    st.caption('Read-only market metadata. No trading, recommendations or validated exchange signals.')
    slug=st.text_input('Polymarket football event slug (from the event URL)').strip()
    if st.button('Load event metadata'):
        try:
            response=requests.get('https://gamma-api.polymarket.com/events',params={'slug':slug},timeout=20)
            response.raise_for_status(); events=response.json()
            if not isinstance(events,list) or not events: st.warning('No matching event returned.'); return
            event=next((e for e in events if e.get('slug')==slug),None)
            if not event: st.warning('Exact slug not found.'); return
            st.subheader(event.get('title','Event'))
            st.caption('Retrieved UTC: '+datetime.now(timezone.utc).isoformat())
            rows=[]
            for m in event.get('markets',[]):
                rows.append({'Market':m.get('question'),'Status': 'Closed' if m.get('closed') else 'Open / verify', 'Reported volume':m.get('volume'),'Reported liquidity':m.get('liquidity'),'End date':m.get('endDate')})
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.caption('Provider-reported cumulative volume and liquidity are different measures. Neither establishes trader skill or future results. Confirm event identity and settlement rules independently.')
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            st.error('Provider response unavailable or invalid. No data inferred. '+str(exc))
    st.info('Kalshi, additional bookmakers, individual trades, xG and lineup feeds are not connected in this preview. Historical validation remains separate.')

if __name__=='__main__': main()
