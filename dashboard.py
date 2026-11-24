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
    st.set_page_config(page_title='Football Intelligence V3.1',layout='wide')
    if st.sidebar.radio('Workspace', ['Historical backtest','Market research']) == 'Market research':
        market_research()
        return
    st.title('Football Intelligence V3.1 — Historical backtest')
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
    st.subheader('Match probabilities'); st.dataframe(r,hide_index=True); st.download_button('Download historical results',r.to_csv(index=False),'v2_results.csv','text/csv')
    with st.expander('Original odds movement diagnostic (different sample)'):
        st.json(s); st.dataframe(old)
    st.warning('Research only. No validated signal threshold. Scores are modal estimates, not guarantees. No xG, injuries, exchange trades or live odds integrated yet.')


def market_research():
    import requests, json
    from datetime import datetime, timezone
    from urllib.parse import urlparse
    st.title('Market research — V3.1')
    st.caption('Read-only provider data. Quotes may be delayed; no validated trading signals.')
    def get(url, params):
        r=requests.get(url,params=params,timeout=20); r.raise_for_status(); return r.json()
    def arr(v):
        return json.loads(v) if isinstance(v,str) else (v or [])
    def fmt(v):
        try: return f'{float(v):,.2f}'
        except (ValueError,TypeError): return 'Unavailable'
    def pct(v):
        try: return f'{float(v):.1%}'
        except (ValueError,TypeError): return 'Unavailable'
    source=st.radio('Provider',['Polymarket','Kalshi'],horizontal=True)
    if source=='Polymarket':
        raw=st.text_input('Event URL or exact slug').strip()
        slug=urlparse(raw).path.rstrip('/').split('/')[-1] if '://' in raw else raw
        if st.button('Load event metadata'):
            try:
                events=get('https://gamma-api.polymarket.com/events',{'slug':slug})
                event=next((e for e in events if e.get('slug')==slug),None)
                if not event: st.warning('Exact event not found.'); return
                st.session_state['poly_event']=(event,datetime.now(timezone.utc).isoformat())
            except (requests.RequestException,ValueError,TypeError) as exc: st.error(str(exc)); return
        if 'poly_event' in st.session_state:
            event,stamp=st.session_state['poly_event']; st.subheader(event.get('title','Event')); st.caption('Retrieved UTC: '+stamp)
            for m in event.get('markets',[]):
                with st.container(border=True):
                    st.markdown('**'+str(m.get('question','Market'))+'**')
                    st.caption('Closed' if m.get('closed') else 'Open / verify')
                    outcomes=arr(m.get('outcomes')); prices=arr(m.get('outcomePrices'))
                    for i,o in enumerate(outcomes): st.write(f'{o}: {pct(prices[i] if i<len(prices) else None)}')
                    st.write('Best bid / ask (provider metadata):',fmt(m.get('bestBid')), '/',fmt(m.get('bestAsk')))
                    st.write('Reported volume:',fmt(m.get('volume')),' | Liquidity:',fmt(m.get('liquidity')))
            st.caption('Outcome prices are provider-reported snapshots, not guaranteed executable quotes. Volume includes both sides.')
    else:
        ticker=st.text_input('Exact Kalshi event ticker').strip().upper()
        if st.button('Load Kalshi event'):
            try:
                data=get('https://api.elections.kalshi.com/trade-api/v2/events/'+requests.utils.quote(ticker,safe=''),{'with_nested_markets':'true'})
                st.session_state['kalshi_event']=(data,datetime.now(timezone.utc).isoformat())
            except (requests.RequestException,ValueError,TypeError) as exc: st.error(str(exc)); return
        if 'kalshi_event' in st.session_state:
            data,stamp=st.session_state['kalshi_event']; event=data.get('event',{}); st.subheader(event.get('title','Event')); st.caption('Retrieved UTC: '+stamp)
            markets=event.get('markets') or data.get('markets') or []
            if not markets: st.warning('No nested markets returned.')
            for m in markets:
                with st.container(border=True):
                    st.markdown('**'+str(m.get('title','Market'))+'**'); st.caption(str(m.get('status','Unknown')))
                    for key in ['yes_bid','yes_ask','no_bid','no_ask','last_price']:
                        v=m.get(key+'_dollars'); legacy=m.get(key)
                        if v is None and legacy is not None: v=float(legacy)/100
                        st.write(key.replace('_',' ').title()+': '+pct(v))
                    st.write('Reported volume (contracts):',m.get('volume_fp',m.get('volume','Unavailable')))
    st.info('Provider snapshots only. Automatic discovery, trade history, cross-provider matching, additional bookmakers and lineup feeds are not yet implemented.')

if __name__=='__main__': main()
