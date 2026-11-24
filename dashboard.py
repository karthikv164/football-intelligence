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
    st.set_page_config(page_title='Football Intelligence V3.5',layout='wide')
    workspace=st.sidebar.radio('Workspace', ['Historical backtest','Market research','Global bookmakers'])
    if workspace=='Global bookmakers':
        global_market()
        return
    if workspace=='Market research':
        market_research()
        return
    st.title('Football Intelligence V3.5 — Historical backtest')
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
    st.title('Market research — V3.5')
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
    source=st.radio('Provider',['Unified search','Polymarket','Kalshi'],horizontal=True)
    if source=='Unified search':
        unified_search()
        return
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

def unified_search():
    import requests, re, json
    from datetime import datetime, timezone
    from concurrent.futures import ThreadPoolExecutor
    from difflib import SequenceMatcher
    P='https://gamma-api.polymarket.com'
    K='https://api.elections.kalshi.com/trade-api/v2'
    def get(url,params):
        r=requests.get(url,params=params,timeout=15); r.raise_for_status(); return r.json()
    def stamp(): return datetime.now(timezone.utc).isoformat()
    def norm(s): return ' '.join(re.findall(r'[a-z0-9]+',str(s).lower()))
    def score(title,query):
        t=norm(title); q=norm(query); words=q.split()
        return sum(w in t.split() for w in words)/max(1,len(words)) + .2*SequenceMatcher(None,t,q).ratio()
    def poly_search(q):
        d=get(P+'/public-search',{'q':q,'limit_per_type':30}); return d.get('events',[])
    def kalshi_search(q):
        # Public events listing is paginated; search a bounded sample, not all events.
        rows=[]; cursor=None
        for _ in range(8):
            params={'limit':200,'status':'open','series_ticker':series}
            if cursor: params['cursor']=cursor
            d=get(K+'/events',params); rows.extend(d.get('events',[])); cursor=d.get('cursor')
            if not cursor: break
        return rows
    q=st.text_input('Match or team name',placeholder='St Louis vs Los Angeles')
    series=st.text_input('Kalshi football series ticker',value='KXMLSGAME',help='Change for other leagues. Discovery scans up to 1,600 open events in this series.') .strip().upper()
    if st.button('Find matches',type='primary') and q.strip():
        results={}; errors={}
        with st.spinner('Searching both providers…'), ThreadPoolExecutor(max_workers=2) as pool:
            jobs={'Polymarket':pool.submit(poly_search,q),'Kalshi':pool.submit(kalshi_search,q)}
            for name,f in jobs.items():
                try: results[name]=sorted(f.result(),key=lambda e:score(e.get('title',''),q),reverse=True)[:15]
                except Exception as exc: errors[name]=str(exc); results[name]=[]
        st.session_state['discovery32']=(q,series,results,errors)
        st.session_state.pop('comparison32',None)
    if 'discovery32' not in st.session_state: return
    oldq,olds,results,errors=st.session_state['discovery32']
    st.caption('Search results for '+oldq+' / '+olds+'. Select the same fixture manually; names alone do not verify settlement rules.')
    selected={}
    for name,key in [('Polymarket','slug'),('Kalshi','event_ticker')]:
        if name in errors: st.warning(name+': '+errors[name])
        options={e[key]:e for e in results.get(name,[]) if e.get(key)}
        selected[name]=st.selectbox(name+' event',[None]+list(options),format_func=lambda x: 'Select event' if x is None else str(options[x].get('title',''))+' | '+x,key='select32'+name)
    confirmed=st.checkbox('I checked that both events refer to the same fixture and regulation-time settlement rules.')
    if st.button('Retrieve both snapshots',disabled=not all(selected.values())):
        def load(name,ident):
            if name=='Polymarket':
                d=get(P+'/events',{'slug':ident}); e=next(x for x in d if x.get('slug')==ident); return e,e.get('markets',[]),stamp()
            d=get(K+'/events/'+requests.utils.quote(ident,safe=''),{'with_nested_markets':'true'}); e=d.get('event',{}); return e,e.get('markets') or d.get('markets') or [],stamp()
        snapshots={}; errors={}
        with st.spinner('Retrieving provider snapshots…'), ThreadPoolExecutor(max_workers=2) as pool:
            jobs={n:pool.submit(load,n,i) for n,i in selected.items()}
            for n,f in jobs.items():
                try: snapshots[n]=f.result()
                except Exception as exc: errors[n]=str(exc)
        st.session_state['comparison32']=(dict(selected),snapshots,errors)
    if 'comparison32' not in st.session_state: return
    ids,snapshots,errors=st.session_state['comparison32']
    if ids!=selected: st.info('Selection changed. Retrieve fresh snapshots.'); return
    def num(x):
        try:
            v=float(x); return v if math.isfinite(v) and 0<=v<=1 else None
        except (TypeError,ValueError): return None
    def kp(m,k):
        v=m.get(k+'_dollars'); old=m.get(k)
        return num(v if v is not None else float(old)/100 if old is not None else None)
    def arr(x):
        try: return json.loads(x) if isinstance(x,str) else x or []
        except ValueError: return []
    def fmt(v): return 'Unavailable' if v is None else f'{v:.1%}'
    quotes={}
    for n,err in errors.items(): st.error(n+': '+err)
    for n,(e,markets,ts) in snapshots.items():
        st.subheader(n+' — '+str(e.get('title',''))); st.caption('Retrieved UTC: '+ts)
        quotes[n]={}
        for i,m in enumerate(markets):
            title=str(m.get('question') or m.get('title') or 'Market'); label=f'{i+1}. {title}'
            if n=='Polymarket':
                bid=num(m.get('bestBid')); ask=num(m.get('bestAsk')); outcomes=arr(m.get('outcomes')); prices=arr(m.get('outcomePrices')); reported=next((num(prices[j]) for j,o in enumerate(outcomes) if str(o).lower()=='yes' and j<len(prices)),None); active=not m.get('closed') and m.get('active') is True and m.get('acceptingOrders') is True
            else:
                bid=kp(m,'yes_bid'); ask=kp(m,'yes_ask'); reported=kp(m,'last_price'); active=m.get('status')=='active'
            mid=(bid+ask)/2 if active and bid is not None and ask is not None and 0<bid<=ask<1 else None
            quotes[n][label]=mid
            with st.container(border=True):
                st.write(title); st.write('YES bid / ask: '+fmt(bid)+' / '+fmt(ask)); st.write('Reported YES / last: '+fmt(reported)); st.write('Active quote midpoint: '+fmt(mid)); st.caption('Reported volume ('+('contracts' if n=='Kalshi' else 'provider units')+'): '+str(m.get('volume_fp',m.get('volume','Unavailable'))))
    if len(quotes)==2 and all(quotes.values()):
        st.subheader('Compare equivalent markets')
        a=st.selectbox('Polymarket outcome',[None]+list(quotes['Polymarket']),format_func=lambda x:x or 'Choose outcome'); b=st.selectbox('Kalshi outcome',[None]+list(quotes['Kalshi']),format_func=lambda x:x or 'Choose outcome')
        same=st.checkbox('These two outcomes have equivalent definitions and settlement rules.')
        if confirmed and same and a and b:
            x=quotes['Polymarket'][a]; y=quotes['Kalshi'][b]
            if x is None or y is None: st.warning('Comparable active bid/ask quotes unavailable.')
            else:
                gap=abs(x-y)*100; st.metric('Midpoint difference',f'{gap:.1f} percentage points'); st.write('Polymarket: '+fmt(x)+' | Kalshi: '+fmt(y))
                if gap>=5: st.warning('Large discrepancy: investigate quote freshness, match state and settlement rules. Not an arbitrage or betting signal.')
    st.caption('Concurrent requests do not guarantee synchronized exchange quotes. Retrieval time is not quote time. Discovery is bounded and may miss fixtures. No live score feed or bookmaker coverage yet.')
    trade_research(snapshots, ids)

def trade_research(snapshots, ids):
    import requests
    from datetime import datetime, timezone
    st.subheader('Public trade history — research')
    st.caption('Bounded recent trade sample. No inference of trader skill or net positions. Trade timestamps are provider supplied.')
    options={}
    for provider,(_,markets,_) in snapshots.items():
        for m in markets:
            ident=m.get('conditionId') if provider=='Polymarket' else m.get('ticker')
            if ident: options[(provider,ident)]=str(m.get('question') or m.get('title') or ident)
    if not options: return
    selected=st.selectbox('Trade market',list(options),format_func=lambda k:k[0]+' — '+options[k])
    def number(v):
        try:
            x=float(v); return x if math.isfinite(x) else None
        except (ValueError,TypeError): return None
    pages=st.selectbox('Maximum pages (500 trades per page)',[1,2,5,10])
    if st.button('Load recent public trades'):
        provider,ident=selected; rows=[]; error=None
        try:
            trades=[]; cursor=None; seen=set()
            for _ in range(pages):
                if provider=='Polymarket':
                    url='https://data-api.polymarket.com/v2/trades'
                    params={'condition':ident,'limit':500,'taker_only':'true'}
                else:
                    url='https://api.elections.kalshi.com/trade-api/v2/markets/trades'
                    params={'ticker':ident,'limit':500}
                if cursor: params['cursor']=cursor
                r=requests.get(url,params=params,timeout=20); r.raise_for_status(); payload=r.json()
                batch=payload['data'] if provider=='Polymarket' else payload['trades']
                if not isinstance(batch,list): raise ValueError('Unexpected trade response')
                trades.extend(batch)
                cursor=(payload.get('pagination') or {}).get('next_cursor') if provider=='Polymarket' else payload.get('cursor')
                if not cursor: break
                if cursor in seen: raise ValueError('Repeated pagination cursor')
                seen.add(cursor)
            for t in trades:
                if provider=='Polymarket':
                    price=number(t.get('price')); size=number(t.get('size')); side=t.get('side'); outcome=t.get('outcome'); ts=pd.to_datetime(t.get('timestamp'),unit='s',utc=True,errors='coerce'); yes=price if str(outcome).lower()=='yes' else 1-price if price is not None and str(outcome).lower()=='no' else None
                else:
                    yes=number(t.get('yes_price_dollars')); old=number(t.get('yes_price'))
                    if yes is None and old is not None: yes=old/100
                    size=number(t.get('count_fp',t.get('count'))); side=t.get('taker_side'); outcome=side; ts=pd.to_datetime(t.get('created_time'),utc=True,errors='coerce'); price=yes if side=='yes' else 1-yes if side=='no' and yes is not None else None
                notional=price*size if price is not None and size is not None and 0<=price<=1 and size>=0 else None
                rows.append(dict(provider=provider,market_id=ident,timestamp_utc=ts,outcome=outcome,reported_side=side,price=price,yes_price=yes,contracts=size,estimated_trade_value_usd=notional,trade_id=t.get('trade_id',t.get('transaction_hash',t.get('transactionHash')))))
        except Exception as exc: error=str(exc)
        st.session_state['trades33']=(selected,rows,error,datetime.now(timezone.utc).isoformat())
    saved=st.session_state.get('trades33')
    if not saved or saved[0]!=selected: return
    _,rows,error,retrieved=saved
    if error: st.error(error); return
    if not rows: st.info('No public trades returned.'); return
    df=pd.DataFrame(rows).sort_values('timestamp_utc'); st.caption('Retrieved UTC: '+retrieved+' | Returned trades: '+str(len(df))+' (bounded sample; not necessarily complete history)')
    st.dataframe(df,hide_index=True)
    st.download_button('Download trade CSV',df.to_csv(index=False),'public_trades.csv','text/csv')
    valid=df.dropna(subset=['timestamp_utc','yes_price'])
    if not valid.empty: st.line_chart(valid.set_index('timestamp_utc')['yes_price'])
    st.caption('Estimated trade value is contracts × reported side price, not net money flow or account exposure. No trader identities or skill rankings are inferred.')
    st.subheader('Transaction activity summary')
    value='estimated_trade_value_usd'
    st.dataframe(df.groupby(['outcome','reported_side'],dropna=False).agg(trades=('price','size'),quantity=('contracts',lambda x:x.sum(min_count=1)),estimated_value=(value,lambda x:x.sum(min_count=1))).reset_index(),hide_index=True)
    st.write('Largest 10 reported fills')
    st.dataframe(df.dropna(subset=[value]).nlargest(10,value),hide_index=True)
    threshold=st.number_input('Large fill threshold (estimated USD)',min_value=1,value=1000,step=100)
    st.write('Fills meeting threshold:',int((df[value]>=threshold).sum()))
    timed=df.dropna(subset=['timestamp_utc']).copy()
    if not timed.empty:
        latest=timed.timestamp_utc.max()
        windows=[]
        for minutes in [1,5,15]:
            subset=timed[timed.timestamp_utc>latest-pd.Timedelta(minutes=minutes)]
            windows.append({'minutes_before_latest_trade':minutes,'sample_covers_window':timed.timestamp_utc.min()<=latest-pd.Timedelta(minutes=minutes),'trades':len(subset),'estimated_value':subset[value].sum(min_count=1)})
        st.dataframe(pd.DataFrame(windows),hide_index=True)
        st.bar_chart(timed.set_index('timestamp_utc')[value].resample('1min').sum(min_count=1))
    st.caption('Large fills do not establish unique traders, net exposure, or predictive skill. Transaction hashes may identify multiple fills; they are not unique fill IDs. Download CSV to retain snapshots; server session storage is temporary.')
    goal=st.text_input('Optional match event time (UTC ISO format, e.g. 2026-11-24T02:00:00Z)')
    if goal:
        g=pd.to_datetime(goal,utc=True,errors='coerce')
        if pd.isna(g): st.warning('Enter a valid UTC timestamp.')
        else:
            before=valid[(valid.timestamp_utc<g)&(valid.timestamp_utc>=g-pd.Timedelta(minutes=5))]; after=valid[(valid.timestamp_utc>=g)&(valid.timestamp_utc<=g+pd.Timedelta(minutes=5))]
            st.write('Five-minute windows around supplied event time (descriptive only)')
            st.dataframe(pd.DataFrame([{'window':n,'trades':len(d),'mean_yes_trade_price':d.yes_price.mean()} for n,d in [('Before',before),('After',after)]]),hide_index=True)

def global_market():
    import os, requests
    from datetime import datetime, timezone
    st.title('Global bookmaker research — V3.5')
    st.caption('Read-only regional odds snapshots; coverage varies. No verified trading signals.')
    key=os.getenv('ODDS_API_KEY','')
    if not key:
        try: key=st.secrets.get('ODDS_API_KEY','')
        except Exception: pass
    if not key:
        st.info('Add ODDS_API_KEY in Streamlit app secrets to enable this workspace.')
        return
    def fetch(path,params=None):
        r=requests.get('https://api.the-odds-api.com/v4/'+path,params={**(params or {}),'apiKey':key},timeout=25)
        if not r.ok: raise ValueError(f'Odds provider HTTP {r.status_code}; check key, coverage or quota.')
        return r.json(),r.headers.get('x-requests-remaining','Unknown')
    if st.button('Discover football leagues'):
        try:
            data,_=fetch('sports'); st.session_state['global_leagues']={x['key']:x['title'] for x in data if x['key'].startswith('soccer_') and not x.get('has_outrights')}
        except Exception as e: st.error(str(e))
    leagues=st.session_state.get('global_leagues',{})
    if not leagues: return
    league=st.selectbox('Football league',list(leagues),format_func=leagues.get)
    regions=st.multiselect('Bookmaker regions',['us','us2','uk','eu','au'],default=['us','uk','eu','au'])
    markets=st.multiselect('Markets',['h2h','totals','spreads'],default=['h2h','totals'])
    st.caption(f'Maximum requested credits: {len(regions)*len(markets)} per snapshot. Regional labels refer to bookmakers, not match locations.')
    if st.button('Retrieve bookmaker snapshot',disabled=not regions or not markets):
        try:
            events,remaining=fetch(f'sports/{league}/odds',{'regions':','.join(regions),'markets':','.join(markets),'oddsFormat':'decimal'})
            now=datetime.now(timezone.utc).isoformat(); rows=[]
            for event in events:
                for book in event.get('bookmakers',[]):
                    for market in book.get('markets',[]):
                        for outcome in market.get('outcomes',[]):
                            rows.append(dict(retrieved_utc=now,event_id=event['id'],league=league,kickoff_utc=event.get('commence_time'),fixture=event['home_team']+' vs '+event['away_team'],bookmaker=book['title'],bookmaker_key=book['key'],quote_updated_utc=market.get('last_update') or book.get('last_update'),market=market['key'],outcome=outcome['name'],line=outcome.get('point'),decimal_odds=outcome.get('price')))
            st.session_state['global_current']=pd.DataFrame(rows);
            if rows:
                history=st.session_state.get('global_history',pd.DataFrame())
                st.session_state['global_history']=pd.concat([history,pd.DataFrame(rows)],ignore_index=True).tail(100000)
            st.session_state['global_remaining']=remaining
        except Exception as e: st.error(str(e) if isinstance(e,ValueError) else 'Provider request failed; retry later.')
    df=st.session_state.get('global_current')
    if df is None: return
    st.write('Remaining API credits:',st.session_state['global_remaining'])
    if df.empty: st.info('No bookmaker quotes returned.'); return
    st.write('Snapshot league:',df.league.iloc[0]); st.write('Bookmakers returned:',df.bookmaker_key.nunique())
    fixture=st.selectbox('Fixture',df.fixture.unique()); selected=df[df.fixture==fixture].copy()
    selected['raw_implied_probability']=1/pd.to_numeric(selected.decimal_odds,errors='coerce').where(lambda x:x>1)
    st.dataframe(selected,hide_index=True)
    st.download_button('Export full bookmaker snapshot',df.to_csv(index=False),'global_odds_snapshot.csv','text/csv')
    global_comparison(selected)
    st.caption('Raw implied probabilities include bookmaker margin. Export snapshots for historical comparison; first observed price is not necessarily opening odds. No bookmaker stakes or money flow are supplied.')



def normalized_quotes(df):
    rows=[]
    for _,g in df[df.market=='h2h'].groupby(['retrieved_utc','event_id','bookmaker_key']):
        g=g.drop_duplicates('outcome'); odds=pd.to_numeric(g.decimal_odds,errors='coerce')
        if len(g)!=3 or not g.outcome.str.lower().eq('draw').any() or not odds.gt(1).all() or not np.isfinite(odds).all(): continue
        raw=1/odds
        for (_,r),v in zip(g.iterrows(),raw/raw.sum()):
            rows.append(dict(retrieved_utc=r.retrieved_utc,event_id=r.event_id,bookmaker=r.bookmaker,bookmaker_key=r.bookmaker_key,outcome=r.outcome,probability=v,overround=raw.sum()-1,quote_updated_utc=r.quote_updated_utc))
    return pd.DataFrame(rows)

def global_comparison(selected):
    st.subheader('Unified global market research')
    q=normalized_quotes(selected)
    if q.empty: st.info('No complete three-way match winner quotes.'); return
    # Collapse regional variants with identical display names; aliases may remain.
    q['book_group']=q.bookmaker.str.lower().str.replace(r'[^a-z0-9]','',regex=True)
    grouped=q.groupby(['book_group','outcome']).probability.mean().reset_index()
    consensus=grouped.groupby('outcome').probability.mean()
    st.caption('Equal-weight mean of margin-normalized three-way quotes; identical normalized bookmaker names grouped. Not a forecast. Exchange commission is not included.')
    st.dataframe((consensus*100).rename('Bookmaker consensus %').reset_index(),hide_index=True)
    st.dataframe(q,hide_index=True)
    st.caption('Provider quote timestamps and retrieval timestamps differ. Check freshness before comparing.')
    from datetime import datetime,timezone
    age=(pd.Timestamp.now(tz='UTC')-pd.to_datetime(q.quote_updated_utc,utc=True,errors='coerce')).dt.total_seconds()
    st.write('Quotes older than 120 seconds:',int((age>120).sum()),'| Missing timestamps:',int(age.isna().sum()))
    saved=st.session_state.get('comparison32')
    if saved:
        ids,snapshots,errors=saved
        st.write('Previously retrieved exchange events')
        for provider,(event,markets,stamp) in snapshots.items(): st.write(provider, event.get('title'),stamp)
        if st.checkbox('I verified these exchange events match this fixture, date and regulation-time rules',key='globalverify'+str(selected.event_id.iloc[0])):
            import json
            for provider,(event,markets,stamp) in snapshots.items():
                options={str(i)+': '+str(m.get('question') or m.get('title')):m for i,m in enumerate(markets)}
                rows=[]
                for outcome,p in consensus.items():
                    label=st.selectbox(provider+' equivalent YES outcome: '+outcome,[None]+list(options),key='globalmap'+provider+outcome+str(selected.event_id.iloc[0]))
                    if label is None: continue
                    m=options[label]
                    try:
                        if provider=='Kalshi':
                            if m.get('status')!='active': continue
                            b=float(m['yes_bid_dollars']) if m.get('yes_bid_dollars') is not None else float(m['yes_bid'])/100
                            a=float(m['yes_ask_dollars']) if m.get('yes_ask_dollars') is not None else float(m['yes_ask'])/100
                        else:
                            if not m.get('active') or m.get('closed') or not m.get('acceptingOrders'): continue
                            b,a=float(m['bestBid']),float(m['bestAsk'])
                        if 0<b<=a<1: rows.append({'outcome':outcome,'bookmaker consensus %':100*p,'exchange midpoint %':50*(a+b),'retrieved UTC':stamp})
                    except (KeyError,ValueError,TypeError): pass
                if rows: st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.caption('Manually mapped outcomes; snapshots may be asynchronous. No combined probability or trading signal.')
    else: st.info('Retrieve both exchange snapshots in Market research → Unified search first, then return here.')
    st.subheader('Observed odds history')
    history=st.session_state.get('global_history',selected)
    uploads=st.file_uploader('Optional earlier bookmaker CSV snapshots',type='csv',accept_multiple_files=True)
    for f in uploads:
        try:
            d=pd.read_csv(f)
            if not set(selected.columns)-{'raw_implied_probability'} <= set(d.columns): raise ValueError('Missing snapshot columns')
            history=pd.concat([history,d],ignore_index=True)
        except Exception: st.warning('Invalid snapshot: '+f.name)
    history=history[history.event_id==selected.event_id.iloc[0]].drop_duplicates(['retrieved_utc','event_id','bookmaker_key','market','outcome','line'])
    h=normalized_quotes(history)
    if not h.empty:
        h['retrieved_utc']=pd.to_datetime(h.retrieved_utc,utc=True,errors='coerce')
        book=st.selectbox('History bookmaker',sorted(h.bookmaker_key.unique()))
        chart=h[h.bookmaker_key==book].pivot_table(index='retrieved_utc',columns='outcome',values='probability')
        if len(chart)>1: st.line_chart(chart)
        else: st.info('Retrieve another snapshot to see movement.')
    st.download_button('Export observed fixture history',history.to_csv(index=False),'fixture_history.csv','text/csv')
    st.caption('Session history is temporary. First observed odds are not opening odds. Manual refresh only to conserve credits.')
    st.subheader('Public transaction research')
    trade=st.file_uploader('Optional public trades CSV for this fixture',type='csv',key='globaltrades')
    if trade and st.checkbox('I verified the trade export belongs to this fixture and market'):
        try:
            t=pd.read_csv(trade); t['estimated_trade_value_usd']=pd.to_numeric(t.estimated_trade_value_usd,errors='coerce'); st.dataframe(t.nlargest(10,'estimated_trade_value_usd'),hide_index=True)
            t['timestamp_utc']=pd.to_datetime(t.timestamp_utc,utc=True,errors='coerce'); st.bar_chart(t.dropna(subset=['timestamp_utc']).set_index('timestamp_utc').estimated_trade_value_usd.resample('5min').sum(min_count=1))
        except Exception: st.warning('Invalid trade export.')
    st.caption('Reported fills are not net money flow or evidence of trader skill.')

if __name__=='__main__': main()

