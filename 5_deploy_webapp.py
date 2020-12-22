import os
import time
import pytz
import json
import logging
import requests
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import streamlit as st
from streamlit import caching
import matplotlib.pyplot as plt
from requests.exceptions import ConnectionError
from src.db import DataBase
from src.utils_stocks import get_curr_price
from src.utils_general import get_yahoo_link
from src.utils_general import get_google_link
from src.utils_general import suppress_stdout
logging.getLogger().setLevel(logging.CRITICAL)
# demo config
demo = 1
f_demo_df_c = os.path.join(os.getcwd(), 'data', 'demo', 'df_c.parquet')
f_demo_df_proba_sm = os.path.join(os.getcwd(), 'data', 'demo', 'df_proba_sm.parquet')
if demo:
    dir_db = os.path.join(os.getcwd(), 'data', 'db')
else:
    dir_db = os.path.join(os.getcwd(), 'data', 'demo')
db = DataBase([], dir_db=dir_db)
# system strings
TEXT_TITLE = '''# Five Minute Midas
### Predicting profitable day trading positions.
---
'''
TEXT_SYMBOLS_FOUND = '### {} of {} selected.'
TEXT_FIG = '''## {} - {}
#### {} - {}
{}
'''
TEXT_LINKS = '''[G-News]({}), [Y-Finance]({})'''
TEXT_BUTTON1 = 'Refresh Cache'
TEXT_BUTTON2 = 'Show Charts - {}'
TEXT_BUTTON3 = 'Show Charts - All Symbols'
TEXT_EXPLAIN = 'Explain'
TEXT_STR_EXPLAIN_1 = 'Latest price: ${}'
TEXT_STR_EXPLAIN_2 = '- At {}, there was {}% chance of profit. Actual profit: {}%'
TEXT_STR_EXPLAIN_3 = '''Price Chart
- Red Line - Volume Weighted Average Price (VWAP)
- Red Point - Bullish RSI Div, current profit *negative*
- Green Point - Bullish RSI Div, current profit *positive*
- Yellow Point - Bullish RSI Div, current profit *zero*'''
TEXT_STR_EXPLAIN_4 = '''RSI Chart (14 Periods)
- Orange Line - *Overbought* Indicator
- Green Line - *Oversold* Indicator'''
TEXT_DESCRIPTION = 'Description'
TEXT_SELECTBOX = '' #'Symbol - Industry - Latest Profit Probability'
TEXT_SLIDER1 = 'Last Profit Probability'
TEXT_SLIDER2 = 'Historical Prediction Range'
TEXT_SIDEBAR_HEADER = '### Advanced Settings'
TEXT_SIDEBAR_INPUT1 = 'Add Symbols (e.g. BYND, IBM)'
TEXT_SIDEBAR_INPUT2 = 'Remove Symbols (e.g. SPOT, BA)'
TEXT_SIDEBAR_INPUT3 = 'Current Positions (e.g. TSLA, 630, BA, 200 )'
TEXT_SIDEBAR_INPUT4 = 'Simulate Time Cutoff (e.g. 0945)'
TEXT_SIDEBAR_RADIO = 'Sort By'
TEXT_SIDEBAR_BUTTON = 'Show Current Profits'
TEXT_SIDEBAR_WARN_DEMO = 'Feature disabled for demo.'
TEXT_SIDEBAR_ERROR = 'Empty or invalid input.'
DATI_OLD = '19930417_0000'
dt_sort_params = {
    'Last Profit Probability':'proba_last',
    'Max Profit Probability':'proba_max',
    'Last Prediction Time':'datetime_last',
    'Symbol':'sym',
}

@st.cache()
def get_df_proba_sm():
    global demo
    global f_demo_df_proba_sm
    if demo:
        time.sleep(.5)
        df_proba_sm_demo = pd.read_parquet(f_demo_df_proba_sm)
        return df_proba_sm_demo
    # api call to get df_proba
    url = 'http://localhost:5000/df_proba_sm'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data='', headers=headers)
    data = json.loads(r.text)
    df = pd.DataFrame(**data)
    for col in ['datetime_last', 'datetime_update']:
        df[col] = pd.to_datetime(df[col]).dt.round('min')
    return df

@st.cache()
def get_df_c(ls_sym, time_str):
    global demo
    global f_demo_df_c
    if demo:
        time.sleep(.5)
        df = pd.read_parquet(f_demo_df_c)
        index = (df['sym'].isin(ls_sym))&(df['datetime'].dt.strftime('%H%M')<=time_str)
        df = df[index]
        return df
    dt_sym = {'ls_sym':ls_sym, 'time_str':time_str}
    url = 'http://localhost:5000/df_c'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, json=json.dumps(dt_sym), headers=headers)
    data = json.loads(r.text)
    df = pd.DataFrame(**data)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df

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

def get_fig(df_c):
    '''Returns price chart and rsi chart for input dataframe
    Args:
        df_c (pandas.Dataframe): datetime, close, sma180, rsi14, vwap, peak_valley, divergence, proba
    Returns:
        fig (pyplot.figure)
    '''
    df = df_c.copy()
    # new columns
    if 'proba' not in df.columns: df['proba'] = -1
    df['proba'] = df['proba'].fillna(0)
    df['period'] = df['Date'] if 'Date' in df.columns else df['datetime'].dt.time.astype('str').str[:5]
    df['close_div'] = np.where(df['proba']>0, df['close'], np.nan)
    df['close_div_profit'] = np.where((df['proba']>0)&(df['profit']>0), df['close'], np.nan)
    df['close_div_loss'] = np.where((df['proba']>0)&(df['profit']<0), df['close'], np.nan)
    df['pv'] = np.where(df['peak_valley']!=0, df['close'], np.nan)
    # setup plot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9,6))
    # top plot - price line plot
    sns.lineplot(data=df, x='period', y='close', ax=axs[0])
    sns.lineplot(data=df, x='period', y='vwap', color='r', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div', color='gold', ax=axs[0])
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
            axs[0].text(x, close, str(round(close, 2)), fontsize=9) # prices at end of line
    # bottom plot - rsi14 line plot
    sns.lineplot(data=df, x='period', y='rsi14', color='k', ax=axs[1])
    axs[1].axhline(y=30, color='g')
    axs[1].axhline(y=70, color='darkorange')
    date_str = df['datetime'].astype('str').to_list()[0][:10]
    axs[1].text(df['period'].max(), df['rsi14'].min(), date_str, horizontalalignment='right')
    # set ticks
    for ax in axs:
        ax.label_outer()
        ax.set_xticks(ax.get_xticks()[::len(ax.get_xticks())//10+1])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    return fig

def get_fig_multi(ls_sym, df_c):
    '''Returns pyplot figure for list of symbols and input dataframe
    Args:
        ls_sym (list of str)
        df_c (pandas.Dataframe): datetime, close, sma180, rsi14, vwap, peak_valley, divergence, proba
    '''
    if not ls_sym: return
    n = (len(ls_sym)+(3-1))//3
    fig, axs = plt.subplots(nrows=n, ncols=3, figsize=(3*4,n*4))
    for i in range(n):
        for j in range(3):
            pos = (i, j)
            if n==1:
                pos = j
            try:
                sym = ls_sym[i*3+j]
                df = df_c[df_c['sym']==sym].copy()
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

def get_df_curr_profit(ls_sym_entry):
    '''Returns dataframe detailing actual profits and price targets for multiple symbols
    Args:
        ls_sym_entry (list of str)
    Returns:
        df_curr_profit (pandas.DataFrame)
    '''
    global demo
    if demo:
        return pd.DataFrame({'Message':[TEXT_SIDEBAR_WARN_DEMO]})
    ls_sym = ls_sym_entry[0::2]
    ls_entry = ls_sym_entry[1::2]
    ls_target = []
    ls_profit = []
    for sym, entry in zip(ls_sym, ls_entry):
        entry = float(entry)
        current = get_curr_price(sym)
        profit = current/entry-1
        ls_target.append(str(round(entry*1.01,3)))
        ls_profit.append(str(round(profit,3)))
    dt_curr_profit = {
        'sym':ls_sym,
        'target':ls_target,
        'profit':ls_profit,
    }
    df_curr_profit = pd.DataFrame(dt_curr_profit)
    df_curr_profit = df_curr_profit.sort_values('profit', ascending=0)
    return df_curr_profit

def get_str_explain(df_c):
    '''Returns multi-line string that explains the features in chart
    Args:
        df_c (pandas.DataFrame)
    Returns:
        str_explain (str)
    '''
    ls_str_explain = []
    # latest price
    latest_price = round(df_c['close'].to_list()[-1], 2)
    ls_str_explain.append(TEXT_STR_EXPLAIN_1.format(latest_price))
    # predictions
    ls_time = df_c[df_c['proba'].notnull()]['datetime'].dt.time.astype('str').str[:5].to_list()
    ls_proba = (df_c[df_c['proba'].notnull()]['proba']*100).astype('str').str[:4].to_list()
    ls_profit = (df_c[df_c['proba'].notnull()]['profit']*100).astype('str').str[:4].to_list()
    for proba, time, profit in zip(ls_proba, ls_time, ls_profit):
        ls_str_explain.append(TEXT_STR_EXPLAIN_2.format(time, proba, profit))
    # chart elements
    for x in ['---', TEXT_STR_EXPLAIN_3, '---', TEXT_STR_EXPLAIN_4]:
        ls_str_explain.append(x)
    str_explain = '\n'.join(ls_str_explain)
    return str_explain

# UI Generation
try:
    st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title='Five Minute Midas')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    c1, c2, c3, c4, c5  = st.beta_columns((1,4,1,4,1))
    # sidebar - add/remove symbols
    st.sidebar.write(TEXT_SIDEBAR_HEADER)
    ls_sym_add = st.sidebar.text_input(TEXT_SIDEBAR_INPUT1).replace(' ','').upper().split(',')
    if ls_sym_add == ['']: ls_sym_add = []
    ls_sym_rem = st.sidebar.text_input(TEXT_SIDEBAR_INPUT2).replace(' ','').upper().split(',')
    time_str = st.sidebar.text_input(TEXT_SIDEBAR_INPUT4).replace(' ','')
    if not time_str: time_str = '9999'
    # sidebar - get sort params
    sort_params = st.sidebar.radio(TEXT_SIDEBAR_RADIO, list(dt_sort_params.keys()))
    sort_params = dt_sort_params[sort_params]
    ascending = 1 if sort_params == 'sym' else 0
    # sidebar - current profit
    ls_sym_entry = st.sidebar.text_input(TEXT_SIDEBAR_INPUT3).replace(' ','').upper().split(',')
    if st.sidebar.button(TEXT_SIDEBAR_BUTTON):
        if len(ls_sym_entry)%2==0:
            df_curr_profit = get_df_curr_profit(ls_sym_entry)
            st.sidebar.write(df_curr_profit)
        else:
            st.sidebar.write(TEXT_SIDEBAR_ERROR)
    with c2:
        st.write(TEXT_TITLE)
        if st.button(TEXT_BUTTON1): caching.clear_cache() # refresh button
        # api call to get proba
        df_proba_sm = get_df_proba_sm()
        date_str = df_proba_sm['datetime_last'].astype('str').to_list()[0][:10]
        # filter params
        tup_proba_last = st.slider(TEXT_SLIDER1, min_value=0, max_value=100, value=(70,100), step=5, format = '%d %%')
        tup_proba_last = tuple(x/100 for x in tup_proba_last)
        ls_past_mins = ['1 min'] + [str(x+2)+' mins' for x in range(8)] + [str(x+1)+' mins' for x in range(10-1, 60, 10)] + ['All']
        past_mins = st.select_slider(TEXT_SLIDER2, ls_past_mins, 'All')
        if past_mins == 'All':
            dati_target_str = DATI_OLD
        else:
            past_mins = past_mins.split()[0]
            past_mins
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
        st.write(TEXT_SYMBOLS_FOUND.format(len(ls_sym), df_proba_sm.shape[0]))
    if ls_sym:
        with c2:
            # single symbol selection
            df_sym = get_df_sym(ls_sym, db)
            df_sym = pd.merge(df_sym, df_proba_sm, how='left', on='sym').sort_values(sort_params, ascending=ascending)
            ls_sym_mod = (df_sym['sym'] + ' - ' + df_sym['ind'] + ' - ' + (df_sym['proba_last']*100).astype('str').str[:4] + '%').to_list()
            sym = st.selectbox(TEXT_SELECTBOX, ls_sym_mod, index=0).split()[0]
            show_single = 1 if st.button(TEXT_BUTTON2.format(sym)) else 0
            show_multi = 1 if st.button(TEXT_BUTTON3) else 0
        with c4:
            if show_single:
                # chart single
                dt_sym = df_sym[df_sym['sym']==sym].reset_index().to_dict('index')[0]
                st.write(TEXT_FIG.format(
                        sym,
                        dt_sym['long_name'],
                        dt_sym['sec'],
                        dt_sym['ind'],
                        get_links(sym)
                    ),
                    unsafe_allow_html=1
                )
                df_c = get_df_c([sym], time_str)
                fig = get_fig(df_c)
                st.pyplot(fig)
                # description
                exp_des = st.beta_expander(TEXT_DESCRIPTION)
                exp_des.write(dt_sym['summary'])
                # explain
                str_explain = get_str_explain(df_c)
                exp_explain = st.beta_expander(TEXT_EXPLAIN)
                exp_explain.write(str_explain)
            elif show_multi:
                # chart multi
                df_c = get_df_c(ls_sym, time_str)
                fig = get_fig_multi(ls_sym, df_c)
                st.pyplot(fig)
                # explain
                exp_explain = st.beta_expander(TEXT_EXPLAIN)
                exp_explain.write(TEXT_STR_EXPLAIN_3)
except ConnectionError:
    st.write(f'Connection error! Try again in a few seconds.')
except Exception as e:
    st.write(f'{type(e).__name__} - {e}')