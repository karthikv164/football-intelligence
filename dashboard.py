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
    st.set_page_config(page_title='Football Intelligence V4.4',layout='wide')
    workspace=st.sidebar.radio('Workspace', ['Market intelligence','Market research','Global bookmakers','Historical backtest','Live intelligence lab'])
    if workspace=='Market intelligence':
        market_first()
        return
    if workspace=='Live intelligence lab':
        live_lab()
        return
    if workspace=='Global bookmakers':
        global_market()
        return
    if workspace=='Market research':
        market_research()
        return
    st.title('Football Intelligence V4.4 — Historical backtest')
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
    st.title('Market research — V4.2')
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
    source=st.radio('Provider',['Goal markets & large trades','Unified search','Polymarket','Kalshi'],horizontal=True)
    if source=='Goal markets & large trades':
        expanded_markets()
        return
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

def market_category(m):
    import re
    title=str(m.get('question') or m.get('title') or '').lower()
    ticker=str(m.get('ticker') or '').upper().split('-')[0]
    if re.search(r'corners|yellow cards|red cards|shots|offsides|fouls|first half|1st half',title) or any(x in ticker for x in ['CORNER','CARD','SHOT','1H','2H']): return 'Other / unclassified'
    if re.search(r'wins? by|winning margin|handicap|spread',title) or 'SPREAD' in ticker:
        return 'Handicap / winning margin'
    if re.search(r'both teams.*scor|btts',title) or 'BTTS' in ticker:
        return 'BTTS'
    if re.search(r'team total|(?:will )?.+ (?:score|scores|to score) (?:over|under|more than|at least|\d)',title) or 'TEAMTOTAL' in ticker:
        return 'Team totals'
    if re.search(r'total goals|goals (?:scored )?in (?:the |this )?(?:match|game)|over/under|\bo/u\b',title) or ticker.endswith('TOTAL'):
        return 'Match goal totals'
    if re.search(r'\bwin\b|\bwins\b|\bdraw\b|\btie\b',title):
        return 'Match winner'
    return 'Other / unclassified'


def expanded_markets():
    import requests, re
    from datetime import datetime, timezone
    from urllib.parse import urlparse
    st.subheader('Goal markets and large trades')
    st.caption('Search related events or enter exact provider identifiers. Verify fixture, date, line and settlement before interpreting trades.')
    provider=st.radio('Exchange',['Polymarket','Kalshi'],horizontal=True,key='expanded_provider')
    query=st.text_input('Fixture search',placeholder='St Louis Los Angeles',key='expanded_query')
    exact=st.text_input('Exact event slug / URL or Kalshi event ticker',key='expanded_exact')
    ticker=st.text_input('Optional exact Kalshi market ticker',key='expanded_ticker') if provider=='Kalshi' else ''
    def get(url,params):
        r=requests.get(url,params=params,timeout=20); r.raise_for_status(); return r.json()
    if st.button('Discover goal markets',type='primary'):
        events=[]; errors=[]
        P='https://gamma-api.polymarket.com'; K='https://api.elections.kalshi.com/trade-api/v2'
        def attempt(fn):
            try: fn()
            except Exception as e: errors.append(str(e))
        if provider=='Polymarket':
            slug=urlparse(exact).path.rstrip('/').split('/')[-1] if '://' in exact else exact.strip()
            if slug: attempt(lambda:events.extend(get(P+'/events',{'slug':slug})))
            if query.strip():
                for term in [query,query+' totals',query+' goals']:
                    attempt(lambda:events.extend(get(P+'/public-search',{'q':term,'limit_per_type':30}).get('events',[])))
        else:
            def load_event(t):
                d=get(K+'/events/'+requests.utils.quote(t,safe=''),{'with_nested_markets':'true'}); e=d.get('event',{}); e['markets']=e.get('markets') or d.get('markets') or []; events.append(e)
            if exact.strip(): load_id=exact.strip().upper(); attempt(lambda:load_event(load_id))
            if ticker.strip():
                def load_market():
                    m=get(K+'/markets/'+requests.utils.quote(ticker.strip().upper(),safe=''),{}).get('market',{}); events.append({'title':m.get('event_ticker'),'markets':[m]})
                attempt(load_market)
            if query.strip():
                # Discover related series from the provider catalog rather than guessing tickers.
                def search_series():
                    series=get(K+'/series',{'category':'Sports'}).get('series',[])
                    candidates=[x for x in series if any(w in str(x.get('title','')).lower() for w in ['soccer','mls','football']) and any(w in str(x.get('title','')).lower() for w in ['goal','total','score','mls'])]
                    words=set(re.findall(r'[a-z0-9]+',query.lower()))-{'vs','fc','sc','the'}
                    for x in candidates[:20]:
                        def scan():
                            d=get(K+'/events',{'series_ticker':x['ticker'],'status':'open','with_nested_markets':'true','limit':200})
                            for e in d.get('events',[]):
                                title=set(re.findall(r'[a-z0-9]+',str(e.get('title','')).lower()))
                                if len(words & title)>=min(2,len(words)) and words: events.append(e)
                        attempt(scan)
                attempt(search_series)
        st.session_state['expanded36']=(provider,events,errors,datetime.now(timezone.utc).isoformat())
    saved=st.session_state.get('expanded36')
    if not saved or saved[0]!=provider: return
    _,events,errors,stamp=saved
    for e in errors: st.warning(e)
    markets={}
    for e in events:
        for m in e.get('markets',[]):
            ident=m.get('conditionId') if provider=='Polymarket' else m.get('ticker')
            if ident: markets[ident]=(e,m)
    if not markets: st.info('No markets found. Try the exact goal-market event identifier.'); return
    def label(k):
        e,m=markets[k]; return str(m.get('question') or m.get('title') or k)+' | '+str(e.get('title',''))+' | '+k
    # Keep ambiguous markets visible but never silently treat them as match totals.
    inventory=[]
    for ident,(event,market) in markets.items():
        inventory.append({'category':market_category(market),'market':market.get('question') or market.get('title'),'event':event.get('title'),'event_id':market.get('event_ticker') or event.get('slug'),'market_id':ident,'status':market.get('status'),'bid':market.get('yes_bid_dollars',market.get('bestBid')),'ask':market.get('yes_ask_dollars',market.get('bestAsk'))})
    st.subheader('Discovered market overview')
    st.caption('One provider snapshot. Verify event identifier, fixture date and settlement rules. Quotes are not synchronized or forecasts.')
    inv=pd.DataFrame(inventory)
    st.dataframe(inv,hide_index=True)
    st.download_button('Export discovered markets',inv.to_csv(index=False),'market_inventory.csv','text/csv')
    event_ids=list(dict.fromkeys(str(x['event_id']) for x in inventory if x['event_id']))
    event_filter=st.selectbox('Exact event identifier (optional)', ['All discovered events']+event_ids)
    category=st.selectbox('Market category',['Match goal totals','BTTS','Team totals','Handicap / winning margin','Match winner','Other / unclassified','All markets'])
    keys=[x['market_id'] for x in inventory if (category=='All markets' or x['category']==category) and (event_filter=='All discovered events' or str(x['event_id'])==event_filter)]
    if not keys: st.info('No goal markets returned. Try All markets or a related event slug/ticker.'); return
    k=st.selectbox('Select exact market',keys,format_func=label)
    e,m=markets[k]; st.caption('Retrieved UTC: '+stamp); st.write(label(k)); st.caption('Market classification: '+market_category(m))
    if provider=='Polymarket': st.write('Reported bid / ask:',m.get('bestBid'),m.get('bestAsk'))
    else: st.write('Reported YES bid / ask:',m.get('yes_bid_dollars',m.get('yes_bid')),m.get('yes_ask_dollars',m.get('yes_ask')))
    trade_research({provider:(e,[m],stamp)},{provider:k})

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
                    url='https://data-api.polymarket.com/trades'
                    params={'market':ident,'limit':500,'takerOnly':'true','offset':len(trades)}
                else:
                    url='https://api.elections.kalshi.com/trade-api/v2/markets/trades'
                    params={'ticker':ident,'limit':500}
                if cursor: params['cursor']=cursor
                r=requests.get(url,params=params,timeout=20); r.raise_for_status(); payload=r.json()
                batch=payload if provider=='Polymarket' else payload['trades']
                if not isinstance(batch,list): raise ValueError('Unexpected trade response')
                trades.extend(batch)
                if provider=='Polymarket':
                    if len(batch)<500: break
                    continue
                cursor=payload.get('cursor')
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
    large=df[df[value]>=threshold].sort_values('timestamp_utc',ascending=False)
    st.metric('Large fills in retrieved sample',len(large))
    st.dataframe(large[['timestamp_utc','outcome','reported_side','price',value]],hide_index=True)
    with st.expander('All reported trades'):
        st.dataframe(df,hide_index=True)
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
            st.write('Five-minute windows around supplied event time (descriptive only)'); st.caption('Pre-event timestamps do not prove advance knowledge: score feeds and exchange clocks may differ. Partial windows are not comparable.')
            st.dataframe(pd.DataFrame([{'window':n,'full_time_span_covered':bool(timed.timestamp_utc.min() <= (g-pd.Timedelta(minutes=5) if n=='Before' else g) and timed.timestamp_utc.max() >= (g if n=='Before' else g+pd.Timedelta(minutes=5))),'trades':len(d),'mean_yes_trade_price':d.yes_price.mean(),'estimated_value_usd':d[value].sum(min_count=1),'large_fills':int((d[value]>=threshold).sum())} for n,d in [('Before',before),('After',after)]]),hide_index=True)

def global_market():
    import os, requests
    from datetime import datetime, timezone
    st.title('Global bookmaker research — V4.2')
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
    totals=selected[selected.market=='totals'].copy()
    if not totals.empty:
        st.subheader('Goal totals — bookmaker consensus')
        rows=[]
        for (book,line),g in totals.groupby(['bookmaker_key','line']):
            g=g.drop_duplicates('outcome'); odds=pd.to_numeric(g.decimal_odds,errors='coerce')
            if set(g.outcome.str.lower())!={'over','under'} or not odds.gt(1).all() or not np.isfinite(odds).all(): continue
            raw=1/odds
            rows.append({'line':line,'book':str(g.bookmaker.iloc[0]).lower(),'over':float((raw/raw.sum())[g.outcome.str.lower()=='over'].iloc[0])})
        if rows:
            a=pd.DataFrame(rows).groupby(['line','book']).over.mean().reset_index().groupby('line').over.agg(['mean','count']).reset_index()
            a['Over %']=100*a['mean']; a['Under %']=100-a['Over %']; st.dataframe(a[['line','Over %','Under %','count']],hide_index=True)
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


def live_lab():
    import requests
    st.title('V4.2 — Live match intelligence lab')
    st.caption('Research workspace. Manual match inputs or optional API-Football feed. Scenario probabilities are not calibrated live forecasts.')
    fixture=st.text_input('Fixture name',key='lab_fixture')
    key=st.secrets.get('API_FOOTBALL_KEY','')
    if key:
        if st.button('Retrieve live fixtures (one provider request)'):
            try:
                r=requests.get('https://v3.football.api-sports.io/fixtures',params={'live':'all'},headers={'x-apisports-key':key},timeout=15); r.raise_for_status(); d=r.json()
                if d.get('errors'): raise ValueError('Provider returned an error')
                st.session_state['lab_live']=d.get('response',[])
            except Exception: st.error('Live feed unavailable; use manual inputs.')
        games=st.session_state.get('lab_live',[])
        if games:
            opts={str(x['fixture']['id']):x for x in games}; ident=st.selectbox('Live fixture',list(opts),format_func=lambda i:opts[i]['teams']['home']['name']+' vs '+opts[i]['teams']['away']['name'])
            x=opts[ident]; st.write(x['teams']['home']['name'],x['goals'],x['teams']['away']['name'],x['fixture']['status']); st.caption('Retrieved snapshot; refresh manually. Verify match identity and status.')
            if st.button('Retrieve statistics and lineups (two requests)'):
                for endpoint in ['fixtures/statistics','fixtures/lineups']:
                    try:
                        r=requests.get('https://v3.football.api-sports.io/'+endpoint,params={'fixture':ident},headers={'x-apisports-key':key},timeout=15); r.raise_for_status(); d=r.json(); st.session_state['lab_'+endpoint]=(ident,d.get('response',[]),d.get('errors'))
                    except Exception: st.warning(endpoint+' unavailable')
            for endpoint in ['fixtures/statistics','fixtures/lineups']:
                saved=st.session_state.get('lab_'+endpoint)
                if saved and saved[0]==ident:
                    with st.expander(endpoint): st.json(saved[1]); st.write('Provider errors:',saved[2])
    else: st.info('Optional live feed: add API_FOOTBALL_KEY to Streamlit secrets. Coverage and quota depend on provider plan. No key needed for manual research.')
    a,b,c=st.columns(3); h=a.number_input('Home goals',0,30,0); v=b.number_input('Away goals',0,30,0); minute=c.number_input('Minute',0,90,52)
    st.caption('Regulation time only; enter score and minute from a verified source. No stoppage time adjustment.')
    lh=st.number_input('Assumed home goals per 90',0.0,8.0,1.5,.1); la=st.number_input('Assumed away goals per 90',0.0,8.0,1.2,.1)
    probs=scenario(h,v,minute,lh,la)
    st.subheader('Goal and result scenarios')
    st.dataframe(pd.DataFrame([{'Market':k,'Scenario probability %':round(p*100,2)} for k,p in probs.items()]),hide_index=True)
    st.warning('Independent Poisson assumptions only. No live xG adjustment, injury adjustment, validated edge or recommended wager.')
    st.subheader('Cross-provider comparison — verified manual quotes')
    market=st.selectbox('Exact market',list(probs)); rows=[]
    for provider in ['Kalshi','Polymarket','Bookmaker']:
        val=st.text_input(provider+' implied probability (%)',key='lab_quote_'+provider)
        if val:
            try:
                p=float(val); assert 0<=p<=100; rows.append({'Provider':provider,'Market':market,'Entered probability %':p})
            except Exception: st.warning('Invalid probability for '+provider)
    if rows: st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.caption('Manual inputs only; confirm same fixture, settlement rules, line and timestamp. Not executable prices or a validated edge.')
    st.subheader('Large-fill review')
    f=st.file_uploader('Upload public trades CSV',type='csv',key='lab_trade'); threshold=st.number_input('Large fill USD',1,1000000,1000)
    if f:
        try:
            d=pd.read_csv(f); d['timestamp_utc']=pd.to_datetime(d.timestamp_utc,utc=True,errors='coerce'); d['estimated_trade_value_usd']=pd.to_numeric(d.estimated_trade_value_usd,errors='coerce'); ids=d.market_id.dropna().unique(); mid=st.selectbox('Exact trade market ID',ids); d=d[d.market_id==mid].dropna(subset=['timestamp_utc']); large=d[d.estimated_trade_value_usd>=threshold]; st.metric('Large fills in sample',len(large)); st.dataframe(large.sort_values('timestamp_utc',ascending=False),hide_index=True)
            event=st.text_input('Verified goal time UTC (ISO timestamp)',key='lab_goal')
            if event:
                t=pd.to_datetime(event,utc=True,errors='raise'); w=pd.Timedelta(minutes=5); result=[]
                for name,start,end in [('Before',t-w,t),('After',t,t+w)]:
                    s=d[(d.timestamp_utc>=start)&(d.timestamp_utc<end)]; result.append({'Window':name,'Sample spans window':bool(d.timestamp_utc.min()<=start and d.timestamp_utc.max()>=end),'Fills':len(s),'Large fills':int((s.estimated_trade_value_usd>=threshold).sum())})
                st.dataframe(pd.DataFrame(result),hide_index=True)
            st.caption('Sample coverage is not proof of completeness. Large fills are not unique whales or net exposure.')
        except Exception: st.warning('Invalid or unsupported trade CSV.')
    st.download_button('Export scenario snapshot',pd.DataFrame([{'fixture':fixture,'minute':minute,'home':h,'away':v,'home_rate':lh,'away_rate':la,'market':k,'probability':p,'created_utc':pd.Timestamp.now(tz='UTC').isoformat()} for k,p in probs.items()]).to_csv(index=False),'scenario.csv','text/csv')

def scenario(h,a,minute,lh,la):
    remaining=max(0,90-minute)/90
    def pm(rate): return np.array([math.exp(-rate)*rate**k/math.factorial(k) for k in range(40)])
    m=np.outer(pm(lh*remaining),pm(la*remaining)); m/=m.sum(); i,j=np.indices(m.shape); H=i+h; A=j+a
    out={'BTTS Yes':m[(H>0)&(A>0)].sum(),'Home to score again':m[i>0].sum(),'Away to score again':m[j>0].sum(),'Another goal':m[i+j>0].sum(),'Home win':m[H>A].sum(),'Draw':m[H==A].sum(),'Away win':m[H<A].sum()}
    for line in [1.5,2.5,3.5,4.5,5.5]:
        out[f'Over {line}']=m[H+A>line].sum(); out[f'Under {line}']=1-out[f'Over {line}']
        for label,goals in [('Home',H),('Away',A)]: out[f'{label} over {line}']=m[goals>line].sum()
    out['BTTS No']=1-out['BTTS Yes']; return {k:float(np.clip(p,0,1)) for k,p in out.items()}

def soccer_series(x):
    import re
    t=str(x.get('ticker','')).upper()
    tags=' '.join(map(str,x.get('tags') or [])).lower()
    title=str(x.get('title','')).lower()
    if re.search(r'\b(soccer|association football)\b',tags+' '+title): return True
    return bool(re.match(r'^KX(?:MLS|EPL|UCL|UEL|UECL|LALIGA|LIGUE1|LIGUE2|BUNDESLIGA|SERIEA|SERIEB|BRASILEIRO|LIBERTADORES|SUDAMERICANA|CONCACAF|ITASUPERCUP|FACUP|EFLCUP|EREDIVISIE|LIGAMX|NWSL|WSL|COPAAMERICA|FIFAWORLDCUP|UEFANATIONS|SCOTTISHPREM|PRIMEIRALIGA)',t))

def bookmaker_consensus44(df):
    rows=[]
    if df.empty: return pd.DataFrame()
    for keys,g in df.groupby(['retrieved_utc','event_id','market','line','bookmaker_key'],dropna=False):
        stamp,event,market,line,book=keys
        g=g.drop_duplicates('outcome'); p=pd.to_numeric(g.decimal_odds,errors='coerce')
        names=set(g.outcome.str.lower())
        if market=='h2h': valid=len(g)==3 and 'draw' in names
        elif market=='totals': valid=names=={'over','under'} and pd.notna(line)
        else: continue
        if not valid or not p.gt(1).all() or not np.isfinite(p).all(): continue
        raw=1/p
        for (_,r),v in zip(g.iterrows(),raw/raw.sum()):
            rows.append(dict(retrieved_utc=stamp,event_id=event,fixture=r.fixture,league=r.league,market=market,line=line,outcome=r.outcome,bookmaker=r.bookmaker,bookmaker_key=book,probability=v,quote_updated_utc=r.quote_updated_utc))
    return pd.DataFrame(rows)


def bookmaker_scanner44(horizon):
    import os, requests
    st.subheader('Global bookmaker scanner')
    st.caption('Pre-match quotes across regions. Bookmaker odds do not disclose customer stakes or money flow.')
    key=os.getenv('ODDS_API_KEY','')
    if not key:
        try: key=st.secrets.get('ODDS_API_KEY','')
        except Exception: pass
    if not key: st.info('Existing ODDS_API_KEY required for bookmaker coverage.'); return
    def fetch(path,params):
        r=requests.get('https://api.the-odds-api.com/v4/'+path,params={**params,'apiKey':key},timeout=20)
        if not r.ok: raise ValueError('Odds API HTTP '+str(r.status_code))
        return r.json(),r.headers.get('x-requests-remaining'),r.headers.get('x-requests-last')
    if st.button('Discover worldwide bookmaker leagues',key='bk44discover'):
        try:
            data,remaining,_=fetch('sports',{})
            st.session_state['bk44leagues']={x['key']:x['title'] for x in data if x['key'].startswith('soccer_') and not x.get('has_outrights') and x.get('active',True)}
            st.session_state['bk44remaining']=remaining
        except Exception as e: st.error(str(e) if isinstance(e,ValueError) else 'League discovery failed.')
    leagues=st.session_state.get('bk44leagues',{})
    if not leagues: return
    st.write('Active soccer leagues discovered:',len(leagues))
    selected=st.multiselect('Leagues to scan (empty = all discovered)',list(leagues),format_func=leagues.get,key='bk44league') or list(leagues)
    regions=st.multiselect('Bookmaker regions',['us','uk','eu','au','us2'],default=['eu','uk'],key='bk44regions')
    markets=st.multiselect('Quote markets',['h2h','totals','spreads'],default=['h2h','totals'],key='bk44markets')
    cap=st.selectbox('Maximum credits per scan',[8,16,24,40],key='bk44cap')
    cost=len(regions)*len(markets); count=min(len(selected),cap//cost) if cost else 0
    remaining=st.session_state.get('bk44remaining')
    if remaining is not None:
        try: count=min(count,int(remaining)//cost) if cost else 0
        except ValueError: pass
    st.caption(f'Next batch: up to {count} leagues; estimated maximum {count*cost} credits. Remaining: {remaining or "unknown"}. Manual refresh only.')
    if st.button('Scan next bookmaker batch',disabled=count==0,key='bk44scan'):
        now=pd.Timestamp.now(tz='UTC'); end=now+pd.Timedelta(hours=horizon); rows=[]; report=[]
        offset=st.session_state.get('bk44offset',0)%len(selected); batch=(selected[offset:]+selected[:offset])[:count]
        for league in batch:
            try:
                events,rem,used=fetch('sports/'+league+'/odds',{'regions':','.join(regions),'markets':','.join(markets),'oddsFormat':'decimal','commenceTimeFrom':now.strftime('%Y-%m-%dT%H:%M:%SZ'),'commenceTimeTo':end.strftime('%Y-%m-%dT%H:%M:%SZ')})
                st.session_state['bk44remaining']=rem
                report.append(dict(league=league,events=len(events),credits=used,status='Returned'))
                for e in events:
                    kickoff=pd.to_datetime(e.get('commence_time'),utc=True,errors='coerce')
                    if pd.isna(kickoff) or not now<kickoff<=end: continue
                    for b in e.get('bookmakers',[]):
                        for m in b.get('markets',[]):
                            for o in m.get('outcomes',[]):
                                rows.append(dict(retrieved_utc=now,event_id=e['id'],league=league,kickoff_utc=kickoff,fixture=e['home_team']+' vs '+e['away_team'],bookmaker=b['title'],bookmaker_key=b['key'],quote_updated_utc=m.get('last_update') or b.get('last_update'),market=m['key'],outcome=o['name'],line=o.get('point'),decimal_odds=o.get('price')))
            except Exception as e:
                report.append(dict(league=league,status=str(e) if isinstance(e,ValueError) else 'Request failed')); break
        st.session_state['bk44offset']=offset+len(report)
        st.session_state['bk44report']=report
        if rows:
            st.session_state['bk44history']=pd.concat([st.session_state.get('bk44history',pd.DataFrame()),pd.DataFrame(rows)],ignore_index=True).tail(150000)
    if 'bk44report' in st.session_state: st.dataframe(pd.DataFrame(st.session_state['bk44report']),hide_index=True)
    hist=st.session_state.get('bk44history',pd.DataFrame())
    if hist.empty: return
    now=pd.Timestamp.now(tz='UTC'); df=hist[(hist.kickoff_utc>now)&(hist.kickoff_utc<=now+pd.Timedelta(hours=horizon))].copy()
    if df.empty: st.info('No retained upcoming quotes in this window.'); return
    df=df.sort_values('retrieved_utc').drop_duplicates(['event_id','bookmaker_key','market','outcome','line'],keep='last')
    st.write(f'{df.event_id.nunique()} fixtures · {df.league.nunique()} leagues · {df.bookmaker_key.nunique()} bookmaker identifiers')
    st.dataframe(df.groupby(['bookmaker','bookmaker_key']).agg(fixtures=('event_id','nunique'),quotes=('outcome','size')).reset_index(),hide_index=True)
    st.download_button('Export bookmaker snapshots',hist.to_csv(index=False),'bookmaker_history.csv',key='bk44export')
    fixture=st.selectbox('Inspect bookmaker fixture',df.event_id.unique(),format_func=lambda i:df.loc[df.event_id==i,'fixture'].iloc[0],key='bk44fixture')
    chosen=df[df.event_id==fixture]; st.dataframe(chosen[['bookmaker','market','line','outcome','decimal_odds','quote_updated_utc','retrieved_utc']],hide_index=True)
    # Normalize only complete contemporaneous outcome sets, then keep latest per book and line.
    q=bookmaker_consensus44(hist[hist.event_id==fixture]);
    if not q.empty:
        q['book_group']=q.bookmaker.str.lower().str.replace(r'[^a-z0-9]','',regex=True)
        latest=q.sort_values('retrieved_utc').drop_duplicates(['bookmaker_key','market','line','outcome'],keep='last')
        latest=latest.groupby(['market','line','outcome','book_group'],dropna=False).probability.mean().reset_index()
        consensus=latest.groupby(['market','line','outcome'],dropna=False).probability.agg(['mean','count']).reset_index(); consensus['mean']*=100
        st.caption('Margin-adjusted consensus; latest complete quote per bookmaker, asynchronous snapshots. Count = normalized bookmaker names.')
        st.dataframe(consensus.rename(columns={'mean':'Consensus %','count':'Books'}),hide_index=True)
        b=st.selectbox('Observed movement bookmaker',sorted(q.bookmaker_key.unique()),key='bk44move')
        g=q[q.bookmaker_key==b].copy(); g['series']=g.market+' '+g.line.fillna('').astype(str)+' '+g.outcome
        st.line_chart(g.pivot_table(index='retrieved_utc',columns='series',values='probability'))
    st.caption('BTTS/team totals depend on provider event-level coverage and are not fetched by this batch. No direct Betfair, Matchbook or Smarkets integration; only returned bookmaker identifiers count as covered. Session history is temporary.')


def market_first():
    import requests
    from concurrent.futures import ThreadPoolExecutor
    st.title('Football Intelligence V4.4 — Global market scanner')
    st.caption('Market-first research. Global bookmaker and public exchange research; no new paid feed or live-score dependency.')
    def get(base,path,params):
        r=requests.get(base+path,params=params,timeout=15); r.raise_for_status(); return r.json()
    P='https://gamma-api.polymarket.com'; K='https://api.elections.kalshi.com/trade-api/v2'; D='https://data-api.polymarket.com'
    horizon=st.radio('Upcoming kickoff window (hours)',[24,48,72],horizontal=True,index=2)
    bookmaker_scanner44(horizon)
    st.divider()
    st.subheader('Exchange market scanner — Kalshi and Polymarket')
    q=st.text_input('Optional team / fixture filter')
    budget=st.selectbox('Kalshi series per scan',[12,30,60],index=1)
    if st.button('Scan global football markets',type='primary'):
        rows=[]; errors=[]; now=pd.Timestamp.now(tz='UTC'); events=[]
        with st.spinner('Discovering football series and markets…'):
            try:
                catalog=get(K,'/series',{'category':'Sports'}).get('series',[])
                series=[x for x in catalog if soccer_series(x) and str(x.get('ticker','')).endswith(('GAME','TOTAL','BTTS','TEAMTOTAL','TTOTAL'))]
                series=sorted(series,key=lambda x:str(x.get('ticker')))
                st.session_state['soccer_catalog43']=series
                if len(series)>budget: errors.append(f'Kalshi: scanning {budget} of {len(series)} candidate series; use next batch to expand.')
                offset=st.session_state.get('offset43',0)%max(1,len(series)); batch=(series[offset:]+series[:offset])[:budget]; st.session_state['offset43']=offset+budget
                def scan(x):
                    try:
                        d=get(K,'/events',{'series_ticker':x['ticker'],'status':'open','with_nested_markets':'true','limit':200})
                        return [('Kalshi',e) for e in d.get('events',[])], ([x['ticker']+': page limit reached'] if d.get('cursor') else [])
                    except Exception as e: return [],[x['ticker']+': '+str(e)]
                with ThreadPoolExecutor(max_workers=3) as pool:
                    for ev,err in pool.map(scan,batch): events.extend(ev); errors.extend(err)
            except Exception as e: errors.append('Kalshi catalog: '+str(e))
            try:
                tags=get(P,'/tags',{'limit':100,'offset':0})
                soccer=next((t for t in tags if str(t.get('slug','')).lower() in ('soccer','football-soccer')),None)
                if soccer is None:
                    soccer=get(P,'/tags/slug/soccer',{})
                for offset in (0,100,200):
                    page=get(P,'/events',{'tag_id':soccer['id'],'active':'true','closed':'false','limit':100,'offset':offset,'order':'volume','ascending':'false'})
                    events.extend(('Polymarket',e) for e in page)
                    if len(page)<100: break
                else: errors.append('Polymarket soccer discovery capped at 300 events.')
            except Exception as e: errors.append('Polymarket soccer catalog: '+str(e))
        for provider,e in events:
            for m in e.get('markets') or []:
                title=m.get('question') or m.get('title') or ''; event=e.get('title',''); ident=m.get('conditionId') if provider=='Polymarket' else m.get('ticker')
                if not ident or m.get('closed') or m.get('status') in ['closed','settled','finalized'] or m.get('active') is False: continue
                category=market_category(m)
                if category not in ['Match goal totals','BTTS','Team totals','Match winner']: continue
                kickoff=pd.to_datetime(m.get('gameStartTime') or e.get('gameStartTime') or e.get('startTime'),utc=True,errors='coerce')
                def num(v):
                    try: return float(v)
                    except (TypeError,ValueError): return np.nan
                bid=num(m.get('bestBid') if provider=='Polymarket' else m.get('yes_bid_dollars')); ask=num(m.get('bestAsk') if provider=='Polymarket' else m.get('yes_ask_dollars'))
                rows.append(dict(retrieved_utc=now,provider=provider,event=event,market=title,category=category,market_id=ident,kickoff_utc=kickoff,phase='Upcoming (provider time)' if pd.notna(kickoff) and kickoff>now else 'Started or time unknown',bid=bid,ask=ask,mid=(bid+ask)/2 if 0<=bid<=ask<=1 else np.nan,volume=num(m.get('volumeNum',m.get('volume')) if provider=='Polymarket' else m.get('volume_fp',m.get('volume'))),volume_unit='Provider USD volume' if provider=='Polymarket' else 'Contracts',liquidity=num(m.get('liquidityNum',m.get('liquidity')) if provider=='Polymarket' else m.get('liquidity_dollars'))))
        df=pd.DataFrame(rows)
        if not df.empty:
            df=df.drop_duplicates(['provider','market_id']); st.session_state['history42']=pd.concat([st.session_state.get('history42',pd.DataFrame()),df],ignore_index=True).tail(50000)
        st.session_state['current42']=df; st.session_state['errors42']=errors
    with st.expander('Discovered soccer series'):
        st.dataframe(pd.DataFrame(st.session_state.get('soccer_catalog43',[])),hide_index=True)
    for e in st.session_state.get('errors42',[]): st.warning(e)
    df=st.session_state.get('current42',pd.DataFrame())
    if df.empty: st.info('Search a fixture or select football series. Discovery is bounded; verify sport, fixture and kickoff.'); return
    st.caption('Volume units differ across exchanges; never sum them. Unknown kickoff is not confirmed pre-match. Polymarket bid/ask refers to first outcome; verify outcome and rules.')
    cats=st.multiselect('Market categories',sorted(df.category.unique()),default=sorted(df.category.unique()))
    view=df[df.category.isin(cats)].copy()
    now=pd.Timestamp.now(tz='UTC'); kickoff=pd.to_datetime(view.kickoff_utc,utc=True,errors='coerce')
    include_unknown=st.checkbox('Include unknown kickoff (not verified pre-match)',value=True)
    view=view[((kickoff>now)&(kickoff<=now+pd.Timedelta(hours=horizon))) | (kickoff.isna() & include_unknown)]
    if q.strip(): view=view[(view.event+' '+view.market).str.contains(q.strip(),case=False,regex=False)]
    st.metric('Markets in current view',len(view))
    sort=st.radio('Rank within each exchange',['volume','liquidity'],horizontal=True)
    st.caption('Unknown kickoff markets are not verified upcoming. Scan again to rotate through candidate series.')
    for provider,g in view.groupby('provider'):
        st.subheader(provider); st.dataframe(g.sort_values(sort,ascending=False),hide_index=True)
    st.download_button('Export market inventory',df.to_csv(index=False),'market_inventory.csv')
    history=st.session_state['history42']; st.download_button('Export session snapshots',history.to_csv(index=False),'market_history.csv')
    if view.empty: return
    keys=list(zip(view.provider,view.market_id)); labels={(r.provider,r.market_id):r.market for r in view.itertuples()}
    chosen=st.multiselect('Compare exact markets (verify same outcome and settlement)',keys,format_func=lambda k:k[0]+' — '+labels[k],max_selections=6)
    for p,i in chosen:
        g=history[(history.provider==p)&(history.market_id==i)].sort_values('retrieved_utc'); st.write(p+' — '+labels[(p,i)]); st.line_chart(g.set_index('retrieved_utc')['mid']); st.caption('Observed snapshots, not opening odds or continuous history.')
    selected=st.selectbox('Inspect transactions',keys,format_func=lambda k:k[0]+' — '+labels[k]); p,i=selected
    trade_research({p:({},[{'conditionId':i,'ticker':i,'title':labels[selected]}],'')},{p:i})
    if p=='Polymarket':
        st.subheader('Public wallet research')
        if st.button('Load wallet-level recent fills'):
            try:
                data=get(D,'/trades',{'market':i,'limit':500,'takerOnly':'true'}); w=pd.DataFrame(data); st.session_state['wallet42']=(i,w)
            except Exception as e: st.error(str(e))
        saved=st.session_state.get('wallet42')
        if saved and saved[0]==i:
            w=saved[1]; cols=['proxyWallet','side','outcome','price','size','timestamp']
            if set(cols)<=set(w):
                w=w[cols].copy(); w['estimated_value']=pd.to_numeric(w.price,errors='coerce')*pd.to_numeric(w['size'],errors='coerce'); st.dataframe(w.sort_values('estimated_value',ascending=False)); st.download_button('Export wallet fills',w.to_csv(index=False),'wallet_fills.csv')
                wallet=st.selectbox('Public wallet',w.proxyWallet.dropna().unique())
                if st.button('Load reported closed positions'):
                    try:
                        c=pd.DataFrame(get(D,'/closed-positions',{'user':wallet,'limit':50})); st.dataframe(c); st.caption('Bounded closed positions, not lifetime performance; selection bias and open losses excluded.')
                    except Exception as e: st.error(str(e))
            else: st.info('Wallet fields unavailable.')
    st.caption('Historical validation: use Historical backtest for existing football model tests. Exchange signal validation requires archived pre-match snapshots and settlement labels; not yet established.')


if __name__=='__main__': main()

