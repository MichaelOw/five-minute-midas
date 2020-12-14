import pytz
import json
import logging
import requests
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit import caching
from requests.exceptions import ConnectionError
from src.db import DataBase
from src.utils_stocks import get_df_c
from src.utils_general import get_yahoo_link
from src.utils_general import get_google_link
from src.utils_general import suppress_stdout
logging.getLogger().setLevel(logging.CRITICAL)
dir_db = json.load(open('dir.txt'))['dir_db']
db = DataBase([], dir_db=dir_db)

TEXT_TITLE = '''# Five Minute Midas
### Predicting profitable day trading positions.
---
'''
TEXT_FIG = '''## {} - {}
#### {} - {}
{}
'''
TEXT_LINKS = '''[Google]({}), [Yahoo Finance]({})'''
DATI_OLD = '19930417_0000'

@st.cache()
def get_df_proba():
    # api call to get df_proba
    url = 'http://localhost:5000/proba'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data='', headers=headers)
    data = json.loads(r.text)
    df = pd.DataFrame(**data)
    for col in ['datetime', 'datetime_update']:
        df[col] = pd.to_datetime(df[col]).dt.round('min')
    df_proba = df.copy()
    # create summary of df_proba
    df1 = (df
            .sort_values('proba',ascending=0)
            .drop_duplicates(subset=['sym'], keep='first')
            .rename(columns={'proba':'proba_max'}))
    df1 = df1[['sym', 'proba_max', 'datetime_update']]
    df2 = (df
            .sort_values('datetime',ascending=0)
            .drop_duplicates(subset=['sym'], keep='first')
            .rename(columns={'datetime':'datetime_last', 'proba':'proba_last'}))
    df2 = df2[['sym', 'datetime_last', 'proba_last']]
    df_proba_sm = pd.merge(df1, df2, how='left', on='sym')
    return df_proba, df_proba_sm

def get_links(sym):
    links = (TEXT_LINKS.format(get_google_link(sym), get_yahoo_link(sym)))
    return links

def get_df_sym(ls_sym, db):
    if len(ls_sym)==1: ls_sym = ls_sym + ls_sym #tuples with one element have incompatible trailing comma
    q='''
        SELECT sym, long_name, sec, ind, summary
          FROM stocks
         WHERE sym IN {}
    '''.format(tuple(ls_sym))
    df_sym = pd.read_sql(q, db.conn)
    return df_sym

def get_fig_multi(ls_sym, df_proba, date_str):
    if not ls_sym: return
    live_data, target_profit, target_loss = (1, 0.011, -0.031)
    n = (len(ls_sym)+(3-1))//3
    fig, axs = plt.subplots(nrows=n, ncols=3, figsize=(3*4,n*4))
    for i in range(n):
        for j in range(3):
            pos = (i, j)
            if n==1:
                pos = j
            try:
                sym = ls_sym[i*3+j]
                df = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
                df = pd.merge(df, df_proba[['sym','datetime','proba']], how='left', on=['sym', 'datetime'])
                df['close_div'] = np.where(df['proba']>0, df['close'], np.nan)
                df['close_div_profit'] = np.where((df['proba']>0)&(df['profit']>0), df['close'], np.nan)
                df['close_div_loss'] = np.where((df['proba']>0)&(df['profit']<0), df['close'], np.nan)
                sns.lineplot(data=df, x='datetime', y='close', ax=axs[pos])
                sns.lineplot(data=df, x='datetime', y='vwap', color='r', ax=axs[pos])
                sns.scatterplot(data=df, x='datetime', y='close_div', color='y', ax=axs[pos])
                sns.scatterplot(data=df, x='datetime', y='close_div_loss', color='r', ax=axs[pos])
                sns.scatterplot(data=df, x='datetime', y='close_div_profit', color='lime', ax=axs[pos])
                axs[pos].set_title(sym, fontsize=20)
            except:
                fig.delaxes(axs[pos])
            axs[pos].set(xticks=[])
            axs[pos].set(xlabel=None)
            axs[pos].set(yticks=[])
            axs[pos].set(ylabel=None)
    return fig

def get_fig(df):
    '''Plot the peaks and valleys of [close] value against [datetime]
    Args:
        df (pandas.Dataframe): datetime, close, sma180, rsi14, vwap, peak_valley, divergence, proba
        title (str): Chart title
    '''
    # new columns
    if 'proba' not in df.columns: df['proba'] = -1
    df['proba'] = df['proba'].fillna(0)
    df['period'] = df['Date'] if 'Date' in df.columns else df['datetime'].dt.time.astype('str').str[:5]
    df['close_div'] = np.where(df['proba']>0, df['close'], np.nan)
    df['close_div_profit'] = np.where((df['proba']>0)&(df['profit']>0), df['close'], np.nan)
    df['close_div_loss'] = np.where((df['proba']>0)&(df['profit']<0), df['close'], np.nan)
    df['pv'] = np.where(df['peak_valley']!=0, df['close'], np.nan)
    # setup plot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,6))
    # top plot - price line plot
    sns.lineplot(data=df, x='period', y='close', ax=axs[0])
    sns.lineplot(data=df, x='period', y='vwap', color='r', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div', color='y', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div_loss', color='r', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div_profit', color='lime', ax=axs[0])
    # top plot - profit proba labels
    for i, point in df.iterrows():
        x = point['period']
        y = df['close'].min() #point['close'] * 0.99
        proba = point['proba']
        profit = point['profit']
        close = point['close']
        if proba:
            axs[0].text(x, y, f'{round(proba, 2)}\n({round(profit, 2)})', fontsize=9)
        if i == df.shape[0]-1:
            axs[0].text(x, close, str(round(close, 2)), fontsize=9)
    # text label at end of line
    
    # bottom plot - rsi14 line plot
    sns.lineplot(data=df, x='period', y='rsi14', color='k', ax=axs[1])
    axs[1].axhline(y=30, color='r')
    axs[1].axhline(y=70, color='g')
    # set ticks
    for ax in axs:
        ax.label_outer()
        ax.set_xticks(ax.get_xticks()[::len(ax.get_xticks())//10+1])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    return fig

def generate_chart_single(df_proba, sym, date_str):
    live_data, target_profit, target_loss = (1, 0.011, -0.031)
    df_c = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
    df_c = pd.merge(df_c, df_proba[['sym','datetime','proba']], how='left', on=['sym', 'datetime'])
    fig = get_fig(df_c)
    st.pyplot(fig)

try:
    st.set_page_config(layout='wide')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    c1, c2, c3, c4, c5  = st.beta_columns((1,4,1,4,1))
    # sidebar - add/remove symbols
    ls_sym_add = st.sidebar.text_input('Add symbols (e.g. BYND, IBM)').replace(' ','').upper().split(',')
    ls_sym_rem = st.sidebar.text_input('Remove symbols (e.g. SPOT, BA)').replace(' ','').upper().split(',')
    ls_sym_entry = st.sidebar.text_input('Current positions (e.g. TSLA, 630.5 )').replace(' ','').upper().split(',')
    if ls_sym_add == ['']: ls_sym_add = []
    # sidebar - get sort params
    sort_params = st.sidebar.radio('Sort By', ['proba_last', 'datetime_last', 'sym'])
    ascending = 1 if sort_params == 'sym' else 0
    with c2:
        st.write(TEXT_TITLE)
        if st.button('Refresh'): caching.clear_cache() # refresh button
        # api call to get proba
        df_proba, df_proba_sm = get_df_proba()
        date_str = df_proba['datetime'].astype('str').to_list()[0][:10]
        # filter params
        tup_proba_last = st.slider('Profit probability range', min_value=0.0, max_value=1.0, value=(0.7,1.0), step=0.05)
        ls_past_mins = [str(x+1) for x in range(9)] + [str(x+1) for x in range(10-1, 60, 10)] + ['All']
        past_mins = st.select_slider('Past minutes to show', ls_past_mins, 'All')
        if past_mins == 'All':
            dati_target_str = DATI_OLD
        else:
            dati_target_str = (datetime.datetime.now(tz=pytz.timezone('US/Eastern'))+datetime.timedelta(minutes=-int(past_mins))).strftime('%Y%m%d_%H%M')
        # generate sym multiselect
        index = ((df_proba_sm['proba_last']>=tup_proba_last[0])
                    &(df_proba_sm['proba_last']<=tup_proba_last[1])
                    &(df_proba_sm['datetime_last'].dt.strftime('%Y%m%d_%H%M')>=dati_target_str))
        ls_sym = df_proba_sm[index].sort_values(sort_params, ascending=ascending)['sym'].to_list()
        # df_proba_sm
        ls_col = ['sym', 'datetime_last', 'proba_last']
        # add, remove sym
        ls_sym = list(dict.fromkeys(ls_sym + ls_sym_add)) #add new sym and remove duplicates
        ls_sym = [x for x in ls_sym if x not in ls_sym_rem]
        # chart multiple
        show_summary = st.checkbox('Show multi chart summary', True)
        if show_summary:
            fig = get_fig_multi(ls_sym, df_proba, date_str)
            st.pyplot(fig)
    with c4:
        if ls_sym:
            # single symbol selection
            df_sym = get_df_sym(ls_sym, db)
            df_sym = pd.merge(df_sym, df_proba_sm, how='left', on='sym').sort_values(sort_params, ascending=ascending)
            ls_sym_mod = (df_sym['sym'] + ' - ' + df_sym['ind'] + ' - ' + df_sym['proba_last'].astype('str').str[:4]).to_list()
            sym = st.selectbox('Select one symbol', ls_sym_mod, index=0).split()[0]
            # chart single
            dt_sym = df_sym[df_sym['sym']==sym].reset_index().to_dict('index')[0]
            st.write(TEXT_FIG.format(sym, dt_sym['long_name'], dt_sym['sec'], dt_sym['ind'], get_links(sym)))
            generate_chart_single(df_proba, sym, date_str)
            # description
            exp_des = st.beta_expander('Description')
            exp_des.write(dt_sym['summary'])
except ConnectionError:
    st.exception(f'Connection error! Try again in a few seconds.')
except Exception as e:
    st.exception(f'{type(e).__name__} - {e}')