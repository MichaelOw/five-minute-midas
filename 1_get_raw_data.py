import os
import json
import winsound
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from src.db import DataBase
from src.utils_beeps import beeps
from src.utils_date import add_days
from src.utils_stocks import get_ls_sym
from src.utils_stocks import get_df_prices
from src.utils_stocks import suppress_stdout
from src.utils_general import db_remove_dups_stocks
from src.utils_general import db_remove_dups_prices_m
from src.utils_general import db_remove_dups_prices_d
dir_db = os.path.join(os.getcwd(), 'data', 'db')
dir_db_demo = os.path.join(os.getcwd(), 'data', 'demo')

###############################
# Initialize functions and db #
###############################
def get_df_info(sym):
    '''Returns dataframe containing general info about input symbol
    Args:
        sym (str): e.g.  BYND
    Returns:
        df_info (pandas.DataFrame)
    '''
    dt_info = yf.Ticker(sym).info
    dt_info['timestamp'] = datetime.datetime.now()
    dt_info['sector'] = dt_info.get('sector')
    dt_col = {
        'symbol':'sym',
        'longName':'long_name',
        'sector':'sec',
        'industry':'ind',
        'quoteType':'quote_type',
        'fundFamily':'fund_family',
        'longBusinessSummary':'summary',
        'timestamp':'timestamp',
    }
    dt_info = {key:dt_info.get(key) for key in dt_col}
    df_info = pd.DataFrame([dt_info])
    df_info = df_info.rename(columns=dt_col)
    return df_info

def get_df_prices_m(ls_sym, ls_date_str):
    '''Returns dataframe containing minute-level prices
    of symbols in input list
    Args:
        ls_sym (List of str): e.g. ['BYND', 'IBM']
    Returns:
        df_prices_m (pandas.DataFrame)
    '''
    candles_min = 200
    ls_df = []
    dt_errors = {}
    for i, sym in enumerate(tqdm(ls_sym)):
        try:
            df = get_df_prices(sym, ls_date_str[0], ls_date_str[-1])
            len_unique_dates = len(df['datetime'].dt.date.unique())
            len_candles_per_day = df.shape[0]/len_unique_dates
            if len_unique_dates<len(ls_date_str):
                dt_errors[sym] = f'Skip: Incomplete data {len_unique_dates} less than {len(ls_date_str)}'
            elif len_candles_per_day<candles_min:
                dt_errors[sym] = f'Skip: Not enough candles per day {len_candles_per_day} less than {candles_min}'
            else:
                ls_df.append(df)
        except Exception as e:
            dt_errors[sym] = f'Error: {sym}, {type(e).__name__}: {str(e)}'
    [print(f'{x} - {dt_errors[x]}') for x in dt_errors]
    print('Errors: {}/{} {:.3f})}'.format(len(dt_errors)), len(ls_sym), len(dt_errors)/len(ls_sym))
    df_prices_m = pd.concat(ls_df)
    return df_prices_m

ls_init_str = [
    #prices_m
    '''CREATE TABLE IF NOT EXISTS prices_m(
        sym TEXT
        ,datetime TEXT
        ,open REAL
        ,high REAL
        ,low REAL
        ,adj_close REAL
        ,volume INTEGER
        ,is_reg_hours INTEGER)''',
    
    '''CREATE INDEX IF NOT EXISTS index_prices_m_all
        ON prices_m(sym, date(datetime), is_reg_hours)''',

    '''CREATE INDEX IF NOT EXISTS index_prices_m_date
        ON prices_m(date(datetime))''',

    #prices_d
    '''CREATE TABLE IF NOT EXISTS prices_d(
        sym TEXT
        ,date TEXT
        ,open REAL
        ,high REAL
        ,low REAL
        ,adj_close REAL
        ,volume INTEGER)''',
   
    '''CREATE INDEX IF NOT EXISTS index_prices_d_date
        ON prices_d(sym, date(date))''',
    
    #stocks
    '''CREATE TABLE IF NOT EXISTS stocks(
        sym TEXT
        ,long_name TEXT
        ,sec TEXT
        ,ind TEXT
        ,quote_type TEXT
        ,fund_family TEXT
        ,summary TEXT
        ,timestamp TEXT)''',

    '''CREATE INDEX IF NOT EXISTS index_stocks
        ON stocks(sym, quote_type)''',
    
    #stocks_error
    '''CREATE TABLE IF NOT EXISTS stocks_error(
        sym TEXT)''',
    
    #proba
    '''CREATE TABLE IF NOT EXISTS proba(
        sym TEXT
        ,datetime TEXT
        ,my_index INTEGER
        ,proba REAL
        ,datetime_update TEXT)''',
]

db = DataBase(ls_init_str, dir_db)
db_demo = DataBase(ls_init_str, dir_db_demo)

#################
# Update stocks #
#################
print('1. Update stocks')
ls_sym = get_ls_sym()
q = '''
    SELECT sym FROM STOCKS UNION ALL
    SELECT sym FROM stocks_error
    --SELECT sym FROM stocks WHERE summary IS NOT NULL
'''
ls_sym_exclude = pd.read_sql(q, db.conn)['sym'].to_list()
ls_sym = [x for x in ls_sym if x not in ls_sym_exclude]
# extract and load
ls_df = []
dt_errors = {}
if ls_sym:
    for i, sym in enumerate(tqdm(ls_sym)):
        try:
            if ((i+1)%50==0 or (i+1)==len(ls_sym)) and ls_df: # load to db in batches
                df = pd.concat(ls_df)
                df.to_sql('stocks', db.conn, if_exists='append', index=0)
                ls_df = []
            df_info = get_df_info(sym)
            ls_df.append(df_info)
        except Exception as e:
            dt_errors[sym] = f'Skipped... ({type(e).__name__}: {str(e)})'
            df = pd.DataFrame([{'sym':sym}])
            df.to_sql('stocks_error', db.conn, if_exists='append', index=0)
    # print errors
    if dt_errors:
        [print(f'{x} - {dt_errors[x]}') for x in dt_errors]
        print('Errors: {}/{} {:.3f})}'.format(len(dt_errors)), len(ls_sym), len(dt_errors)/len(ls_sym))
    # remove duplicates
    db_remove_dups_stocks(db)
beeps(1)

###################
# Update prices_m #
###################
print('\n2. Update prices_m')
# get max date present
q = '''
    SELECT DATE(MAX(datetime))
      FROM prices_m
     WHERE sym='IBM'
'''
max_date_str = pd.read_sql(q, db.conn).iloc[0,0]
# get missing dates
df = yf.download('IBM', period='1y', interval='1d', progress=0).reset_index()
df['Date']= df['Date'].astype('str')
ls_date_str = df[df['Date']>max_date_str]['Date'].to_list()
assert ls_date_str, 'No dates to extract!'
print('Extracing prices in ', ls_date_str)
# get ls_sym
q = '''
    SELECT sym
      FROM stocks
     WHERE sec IS NOT NULL
'''
ls_sym = pd.read_sql(q, db.conn)['sym'].to_list()
# extract and load
df_prices_m = get_df_prices_m(ls_sym, ls_date_str)
if not df_prices_m.empty:
    df_prices_m.to_sql('prices_m', db.conn, if_exists='append', index=0)
    db_remove_dups_prices_m(db, ls_date_str[0])
beeps(1)

###################
# Update prices_d #
###################
# get max date present
q = '''
    SELECT DATE(MAX(date))
      FROM prices_d
     WHERE sym='IBM'
'''
max_date_str = pd.read_sql(q, db.conn).iloc[0,0]
# check dates
end = add_days(datetime.datetime.today().strftime('%Y-%m-%d'), 3) #today's date plus 3 days
df = yf.download('IBM', start=max_date_str, end=end, interval='1d', progress=0).reset_index()
df = df[df['Date'].astype('str')>max_date_str]
if not df.empty:
    print(f'Extracting from {max_date_str} plus 1, to {end}')
    # get ls_sym
    q = '''
        SELECT sym
          FROM stocks
         WHERE sec IS NOT NULL
    '''
    ls_sym = pd.read_sql(q, db.conn)['sym'].to_list()
    # extract
    dt_cols = {
        'sym':'sym',
        'Date':'date',
        'Open':'open',
        'High':'high',
        'Low':'low',
        'Adj Close':'adj_close',
        'Volume':'volume',
    }
    count = 0
    count_e = 0
    ls_df = []
    dt_errors = {}
    for i, sym in enumerate(tqdm(ls_sym)):
        count+=1
        try:
            with suppress_stdout():
                df = yf.download(sym, period = '1mo', interval='1d', progress=0).reset_index()
            df['sym'] = sym
            df = df.rename(columns=dt_cols)
            df = df[list(dt_cols.values())]
            ls_df.append(df)
        except Exception as e:
            dt_errors[sym] = f'Error: {type(e).__name__}: {str(e)}'
            count_e+=1
    if ls_df:
        # load
        [print(f'{x} - {dt_errors[x]}') for x in dt_errors]
        print('Errors: {}/{} {:.3f})}'.format(len(dt_errors)), len(ls_sym), len(dt_errors)/len(ls_sym))
        df = pd.concat(ls_df)
        df.to_sql('prices_d', db.conn, if_exists='append', index=0)
        db_remove_dups_prices_d(db, max_date_str)
beeps()