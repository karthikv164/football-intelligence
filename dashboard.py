import streamlit as st
import pandas as pd
from engine import fetch_season,backtest,LEAGUES
st.set_page_config(page_title='Football Intelligence — Research',layout='wide')
st.title('Global Football Intelligence | Research Prototype')
st.caption('Historical odds research only. No live sportsbook access, whale claims, or automatic betting.')
with st.sidebar:
    st.header('Backtest settings')
    season=st.text_input('Season code',value='2627')
    selected=st.multiselect('Leagues',options=list(LEAGUES),default=list(LEAGUES),format_func=lambda x:LEAGUES[x])
    start=st.date_input('From',value=pd.Timestamp('2026-11-16'))
    end=st.date_input('To',value=pd.Timestamp('2026-11-22'))
    threshold=st.slider('Minimum implied-probability movement',0.005,0.15,0.03,0.005)
    go=st.button('Load and analyze',type='primary')
if go:
    if not selected: st.error('Choose at least one league.');st.stop()
    if start>end: st.error('Start date must be before end date.');st.stop()
    with st.spinner('Downloading/caching historical CSVs…'):
        try: df,errors=fetch_season(season,selected)
        except Exception as exc: st.error(str(exc));st.stop()
    signals,summary=backtest(df,str(start),str(end),threshold)
    a,b,c,d=st.columns(4)
    a.metric('Matches',summary['matches_in_period']);b.metric('Signals',summary['qualifying_signals']);c.metric('Win rate',f"{summary['win_rate']:.1%}" if summary['win_rate'] is not None else 'N/A');d.metric('Diagnostic ROI',f"{summary['roi']:.1%}" if summary['roi'] is not None else 'N/A')
    st.warning(summary['important_limitation'])
    if errors: st.info('Some leagues unavailable: '+'; '.join(errors))
    st.subheader('Qualifying historical signals')
    st.dataframe(signals,use_container_width=True,hide_index=True)
    st.download_button('Download signal CSV',signals.to_csv(index=False).encode(),'backtest_signals.csv','text/csv')
    st.subheader('Matches used')
    st.dataframe(df[(df.Date>=pd.Timestamp(start))&(df.Date<=pd.Timestamp(end))][['Date','League','HomeTeam','AwayTeam','FTHG','FTAG','FTR']],use_container_width=True,hide_index=True)
else:
    st.info('Choose leagues and click **Load and analyze** to run the historical test.')
    st.markdown('**Next phases:** timestamped bookmaker odds, public exchange trades, market matching, injury/lineup signals, and genuine out-of-sample evaluation.')
