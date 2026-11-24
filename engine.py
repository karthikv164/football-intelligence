"""Free-first football odds research. Read-only. No wager execution."""
import argparse
import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests

LEAGUES = {'E0':'England Premier League','E1':'England Championship','D1':'Germany Bundesliga','SP1':'Spain La Liga','I1':'Italy Serie A','F1':'France Ligue 1','N1':'Netherlands Eredivisie','P1':'Portugal Primeira Liga','B1':'Belgium Pro League','SC0':'Scotland Premiership'}
BASE = 'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
COLUMNS = ['B365H','B365D','B365A']

def fetch_season(season='2627', leagues=None, cache_dir='data'):
    folder=Path(cache_dir); folder.mkdir(parents=True,exist_ok=True)
    all_frames=[]; failures=[]
    for league in (leagues or list(LEAGUES)):
        path=folder/f'{season}_{league}.csv'
        if not path.exists():
            try:
                response=requests.get(BASE.format(season=season,league=league),timeout=25,headers={'User-Agent':'FootballResearch/0.1'})
                response.raise_for_status()
                if not response.content.strip(): raise ValueError('Empty response')
                path.write_bytes(response.content)
            except Exception as e:
                failures.append(f'{league}: {e}'); continue
        try:
            df=pd.read_csv(path,on_bad_lines='skip',low_memory=False)
            required={'Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'}
            if not required.issubset(df.columns): raise ValueError('Missing expected match columns')
            df['LeagueCode']=league; df['League']=LEAGUES.get(league,league)
            all_frames.append(df)
        except Exception as e: failures.append(f'{league}: {e}')
    if not all_frames: raise RuntimeError('No valid data downloaded. '+ '; '.join(failures))
    combined=pd.concat(all_frames,ignore_index=True)
    combined['Date']=pd.to_datetime(combined['Date'],dayfirst=True,errors='coerce')
    for c in ['FTHG','FTAG']+COLUMNS+['B365CH','B365CD','B365CA','AvgH','AvgD','AvgA','AvgCH','AvgCD','AvgCA']:
        if c in combined: combined[c]=pd.to_numeric(combined[c],errors='coerce')
    combined=combined.dropna(subset=['Date','HomeTeam','AwayTeam','FTHG','FTAG'])
    return combined.sort_values(['Date','League','HomeTeam']).reset_index(drop=True),failures

def implied_probabilities(odds):
    vals=np.array(odds,dtype=float)
    if not np.isfinite(vals).all() or (vals<=1).any(): return None
    inv=1/vals
    return inv/inv.sum()

def odds_triplet(row,closing=False):
    choices=([['AvgCH','AvgCD','AvgCA'],['B365CH','B365CD','B365CA']] if closing else [['AvgH','AvgD','AvgA'],COLUMNS])
    for keys in choices:
        if all(k in row.index and pd.notna(row[k]) for k in keys):
            vals=[float(row[k]) for k in keys]
            if min(vals)>1: return vals, '/'.join(keys)
    return None,None

def backtest(df,start='2026-11-16',end='2026-11-22',min_edge=0.03):
    """Pre-match opening->closing odds movement signal; no future match data used in signal.
    This is a retrospective *closing-price* diagnostic, NOT a tradable strategy unless
    closing quotes were available before the selected wager timestamp.
    """
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    sample=df[(df.Date>=start)&(df.Date<end+pd.Timedelta(days=1))]
    results=[]; skipped=0
    for _,r in sample.iterrows():
        op,osrc=odds_triplet(r); cl,csrc=odds_triplet(r,True)
        if op is None or cl is None: skipped+=1; continue
        p0=implied_probabilities(op); p1=implied_probabilities(cl)
        if p0 is None or p1 is None: skipped+=1; continue
        change=p1-p0; i=int(np.argmax(change))
        if change[i]<min_edge: continue
        actual={'H':0,'D':1,'A':2}.get(str(r.FTR))
        if actual is None: skipped+=1; continue
        # Use closing price as the assumed execution price: best-case diagnostic, not guaranteed fill.
        won=(i==actual); odds=cl[i]
        results.append({'date':r.Date.date().isoformat(),'league':r.League,'home':r.HomeTeam,'away':r.AwayTeam,'score':f'{int(r.FTHG)}-{int(r.FTAG)}','selection':['Home','Draw','Away'][i],'opening_probability':round(float(p0[i]),4),'closing_probability':round(float(p1[i]),4),'probability_move':round(float(change[i]),4),'assumed_closing_odds':round(float(odds),3),'won':won,'units_return':round(odds-1 if won else -1,3),'opening_source':osrc,'closing_source':csrc})
    bets=pd.DataFrame(results)
    n=len(bets); profit=float(bets.units_return.sum()) if n else 0
    summary={'period':f'{start.date()} to {end.date()}','matches_in_period':len(sample),'matches_missing_open_or_close_odds':skipped,'qualifying_signals':n,'wins':int(bets.won.sum()) if n else 0,'win_rate':round(float(bets.won.mean()),4) if n else None,'roi':round(profit/n,4) if n else None,'net_units':round(profit,3),'signal':'largest opening-to-closing implied probability increase','minimum_probability_move':min_edge,'important_limitation':'Closing odds signal is retrospective; do not interpret ROI as achievable without pre-kickoff timestamped quotes.'}
    return bets,summary

def run(season='2627',start='2026-11-16',end='2026-11-22',min_edge=.03,leagues=None):
    import json
    df,failures=fetch_season(season,leagues)
    out=Path('output');out.mkdir(exist_ok=True)
    df.to_csv(out/'matches.csv',index=False)
    bets,summary=backtest(df,start,end,min_edge)
    summary['source']='football-data.co.uk season CSV';summary['failed_leagues']=failures
    bets.to_csv(out/'backtest_signals.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return bets,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--season',default='2627');parser.add_argument('--start',default='2026-11-16');parser.add_argument('--end',default='2026-11-22');parser.add_argument('--min-edge',type=float,default=.03);parser.add_argument('--leagues',nargs='*',choices=list(LEAGUES),default=None)
    a=parser.parse_args();run(a.season,a.start,a.end,a.min_edge,a.leagues)
