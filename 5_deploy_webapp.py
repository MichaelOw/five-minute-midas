'''
💹💰💷💶💴💵💸🤖👩‍💻🧑‍💻👨‍💻📉📈📊
'''
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
dir_db = os.path.join(os.getcwd(), 'data', 'db')
if demo: dir_db = os.path.join(os.getcwd(), 'data', 'demo')
db = DataBase([], dir_db=dir_db)
# system strings
TEXT_PAGE_TITLE = 'Five Minute Midas'
TEXT_TITLE = '''# Five Minute Midas 📈
### Predicting profitable day trading positions for *{}*.
---'''
TEXT_ADVICE = '\n ### Try changing the **Profit Probability.**'
TEXT_SYMBOLS_FOUND = '### {} of {} symbols selected.{}\n---'
TEXT_FIG = '''## {} - {} {}
#### {} - {}
{}
'''
TEXT_FIG_MULTI = '## All Symbols Summary'
TEXT_LINKS = '''[G-News]({}), [Y-Finance]({})'''
TEXT_BUTTON1 = 'Refresh Cache'
TEXT_BUTTON3 = 'or Show All'
TEXT_EXPLAIN = 'Explain'
TEXT_STR_EXPLAIN_1 = 'Latest price: ${}, {} from day before'
TEXT_STR_EXPLAIN_2 = '- At {}, there was {}% chance of profit. Actual profit: {}%'
TEXT_STR_EXPLAIN_3 = '''Price Chart
- Red Line - Volume Weighted Average Price (VWAP)
- Red Point - Bullish RSI Div, current profit *negative*
- Green Point - Bullish RSI Div, current profit *positive*'''
TEXT_STR_EXPLAIN_4 = '''RSI Chart (14 Periods)
- Orange Line - *Overbought* Indicator
- Green Line - *Oversold* Indicator'''
TEXT_DESCRIPTION = 'Company Description'
TEXT_SELECTBOX = '' #'Symbol - Industry - Profit Probability (Latest)'
TEXT_SELECT_DEFAULT = 'Choose a Symbol...'
TEXT_SLIDER1 = 'Profit Probability (Latest)'
TEXT_SLIDER2 = 'Historical Prediction Range'
TEXT_SIDEBAR_HEADER = '### Advanced Settings'
TEXT_SIDEBAR_INPUT1 = 'Add Symbols (e.g. BYND IBM)'
TEXT_SIDEBAR_INPUT2 = 'Remove Symbols (e.g. SPOT BA)'
TEXT_SIDEBAR_INPUT3 = 'Current Positions (e.g. TSLA 630)'
TEXT_SIDEBAR_INPUT4 = 'Simulate Time Cutoff (e.g. 0945)'
TEXT_SIDEBAR_RADIO = 'Sort By'
TEXT_SIDEBAR_BUTTON = 'Show Current Profits'
TEXT_SIDEBAR_WARN_DEMO = 'Feature disabled for demo.'
TEXT_SIDEBAR_ERROR = 'Empty or invalid input.'
TEXT_SIDEBAR_INFO = '''### Information
- See code: [GitHub](https://github.com/MichaelOw/five-minute-midas)
- Developer: [Michael](https://www.linkedin.com/in/michael-ow/)
- Read article: Coming soon!
'''
DATI_OLD = '19930417_0000'
dt_sort_params = {
    'Profit Probability (Latest)':'proba_last',
    'Profit Probability (Max)':'proba_max',
    'Prediction Time (Latest)':'datetime_last',
    'Symbol':'sym',
}

@st.cache()
def get_predictions_summary():
    global demo
    global f_demo_df_proba_sm
    if demo:
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
def get_predictions(ls_sym, time_str):
    global demo
    global f_demo_df_c
    if demo:
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
    bar = st.progress(0.0)
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
                bar.progress((i*3+j+1)/(len(ls_sym)))
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

def get_str_explain(df_c, pchange_str):
    '''Returns multi-line string that explains the features in chart
    Args:
        df_c (pandas.DataFrame)
        pchange_str (str)
    Returns:
        str_explain (str)
    '''
    ls_str_explain = []
    # latest price
    latest_price = round(df_c['close'].to_list()[-1], 2)
    ls_str_explain.append(TEXT_STR_EXPLAIN_1.format(latest_price, pchange_str))
    # predictions
    ls_time = df_c[df_c['proba'].notnull()]['datetime'].dt.time.astype('str').str[:5].to_list()
    ls_proba = (df_c[df_c['proba'].notnull()]['proba']*100).astype('str').str[:4].to_list()
    ls_profit = (df_c[df_c['proba'].notnull()]['profit']*100).astype('str').str[:4].to_list()
    for proba, time, profit in zip(ls_proba, ls_time, ls_profit):
        ls_str_explain.append(TEXT_STR_EXPLAIN_2.format(time, proba, profit))
    # chart elements
    for x in ['---', TEXT_STR_EXPLAIN_3, '---', TEXT_STR_EXPLAIN_4, '---']:
        ls_str_explain.append(x)
    str_explain = '\n'.join(ls_str_explain)
    return str_explain

def get_ls_sym_mod(df_sym, sort_params):
    '''Return nice str list that will be display in selectbox
    Args:
        df_sym (pandas.DataFrame)
        sort_params (str)
    Returns:
        ls_sym_mod (list of str)
    '''
    ls_sym_mod = (df_sym['sym']
                    + ' - '
                    + df_sym['ind']
                    + ' - '
                    + (df_sym['proba_last']*100).astype('str').str[:4]
                    + '%'
                    + np.where(df_sym['proba_last']==df_sym['proba_max']
                        ,''
                        ,' (Max: '+(df_sym['proba_max']*100).astype('str').str[:4]+'%)')
                    ).to_list()
    return ls_sym_mod

def get_sidebar_text_input_list(label):
    '''Template for getting nicely formatted list from text_input widget
    Args:
        label (str): text_input widget label
    Returns:
        ls_str (list of str)
    '''
    ls_str = st.sidebar.text_input(label).replace(',',' ').upper().split(' ')
    ls_str = [x for x in ls_str if x]
    if ls_str == ['']: ls_str = []
    return ls_str

def get_pchange_str(df_c):
    '''Returns nice formatted string of the percentage
    change in price from yesterday's close
    Args:
        df_c (pandas.DataFrame)
    Returns:
        pchange_str (str)
    '''
    close_prev = df_c['prev_close'].values[0]
    close_latest =  df_c['close'].values[-1]
    pchange = (close_latest/close_prev-1)*100
    pchange_color = 'green' if pchange>0 else 'red'
    pchange_sign = '+' if pchange>0 else ''
    pchange_str = '<span style="color:{}">{}{:.2f}%</span>'.format(pchange_color, pchange_sign, pchange)
    return pchange_str

# UI Generation
try:
    # config and columns
    st.set_page_config(initial_sidebar_state='collapsed', page_title=TEXT_PAGE_TITLE) #layout='wide'
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #c1, c2, c3, c4, c5  = st.beta_columns((1,4,1,4,1))
    # sidebar - get sort params
    st.sidebar.write(TEXT_SIDEBAR_HEADER)
    sort_params = st.sidebar.radio(TEXT_SIDEBAR_RADIO, list(dt_sort_params.keys()))
    sort_params = dt_sort_params[sort_params]
    ascending = 1 if sort_params == 'sym' else 0
    # sidebar - add/remove symbols
    ls_sym_add = get_sidebar_text_input_list(TEXT_SIDEBAR_INPUT1)
    ls_sym_rem = get_sidebar_text_input_list(TEXT_SIDEBAR_INPUT2)
    time_str = st.sidebar.text_input(TEXT_SIDEBAR_INPUT4).replace(' ','')
    if not time_str: time_str = '9999'
    # sidebar - current profit
    ls_sym_entry = get_sidebar_text_input_list(TEXT_SIDEBAR_INPUT3)
    if st.sidebar.button(TEXT_SIDEBAR_BUTTON):
        if len(ls_sym_entry)%2==0:
            df_curr_profit = get_df_curr_profit(ls_sym_entry)
            st.sidebar.write(df_curr_profit)
        else:
            st.sidebar.write(TEXT_SIDEBAR_ERROR)
    # sidebar - other information
    st.sidebar.write(TEXT_SIDEBAR_INFO)
    empty_slot1 = st.empty()
    if not demo:
        if st.button(TEXT_BUTTON1): caching.clear_cache() # refresh button
    # api call to get proba
    df_proba_sm = get_predictions_summary()
    date_str = df_proba_sm['datetime_last'].astype('str').to_list()[0][:10]
    empty_slot1.write(TEXT_TITLE.format(date_str))
    # filter params
    tup_proba_last = st.slider(TEXT_SLIDER1, min_value=0, max_value=100, value=(90,100), step=5, format = '%d %%')
    tup_proba_last = tuple(x/100 for x in tup_proba_last)
    ls_past_mins = ['1 min'] + [str(x+2)+' mins' for x in range(8)] + [str(x+1)+' mins' for x in range(10-1, 60, 10)] + ['All']
    if demo:
        past_mins = 'All'
    else:
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
    ls_sym = list(dict.fromkeys(ls_sym + ls_sym_add)) # remove duplicates without order change
    ls_sym = [x for x in ls_sym if x not in ls_sym_rem]
    optional_text = TEXT_ADVICE if len(ls_sym)==0 else ''
    st.write(TEXT_SYMBOLS_FOUND.format(len(ls_sym), df_proba_sm.shape[0], optional_text))
    if ls_sym:
        # single symbol selection
        df_sym = get_df_sym(ls_sym, db)
        df_sym = pd.merge(df_sym, df_proba_sm, how='left', on='sym').sort_values(sort_params, ascending=ascending)
        ls_sym_mod = get_ls_sym_mod(df_sym, sort_params)
        ls_sym_mod = ls_sym_mod + [TEXT_SELECT_DEFAULT]
        sym = st.selectbox(TEXT_SELECTBOX, ls_sym_mod, index=len(ls_sym_mod)-1).split()[0]
        chart_type = 'single'
        if len(ls_sym) <= 30 and st.button(TEXT_BUTTON3):
            chart_type = 'multi'
        # charts
        if chart_type == 'multi':
            # chart multi
            st.write(TEXT_FIG_MULTI)
            df_c = get_predictions(ls_sym, time_str)
            fig = get_fig_multi(ls_sym, df_c)
            st.pyplot(fig)
            # explain
            exp_explain = st.beta_expander(TEXT_EXPLAIN)
            exp_explain.write(TEXT_STR_EXPLAIN_3)
        elif chart_type == 'single' and sym != TEXT_SELECT_DEFAULT.split()[0]:
            # chart single
            dt_sym = df_sym[df_sym['sym']==sym].reset_index().to_dict('index')[0]
            df_c = get_predictions([sym], time_str)
            pchange_str = get_pchange_str(df_c)
            fig = get_fig(df_c)
            st.write(TEXT_FIG.format(
                    sym,
                    dt_sym['long_name'],
                    pchange_str,
                    dt_sym['sec'],
                    dt_sym['ind'],
                    TEXT_LINKS.format(get_google_link(sym), get_yahoo_link(sym))
                ),
                unsafe_allow_html=1
            )
            st.pyplot(fig)
            # explain
            str_explain = get_str_explain(df_c, pchange_str)
            exp_explain = st.beta_expander(TEXT_EXPLAIN)
            exp_explain.write(str_explain, unsafe_allow_html=1)
            # description
            exp_des = st.beta_expander(TEXT_DESCRIPTION)
            exp_des.write(dt_sym['summary'])
except ConnectionError:
    st.write(f'Connection error! Try again in a few seconds.')
except Exception as e:
    st.write(f'{type(e).__name__} - {e}')