date_start = '2020-12-21' #'2020-06-23'

import os
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from tqdm import tqdm
from src.db import DataBase
import matplotlib.pyplot as plt
from src.utils_beeps import beeps
from src.utils_stocks import get_df_c
from src.utils_general import get_df_sym
from src.utils_general import plot_divergences
from src.utils_date import get_ls_date_str_from_db
DIR_DB = os.path.join(os.getcwd(), 'data', 'db')
DIR_TRAIN = os.path.join(os.getcwd(), 'data', 'train')
MSG_DATE_RANGE = 'Creating df_train for date range: {} to {}'
MSG_SAVED = '{} saved!'

db = DataBase([], DIR_DB)
# params
live_data = 0
target_profit = 0.011
target_loss = -0.031
ls_date_str = get_ls_date_str_from_db(date_start, '2022-01-01', db) 
print(MSG_DATE_RANGE.format(ls_date_str[0], ls_date_str[-1]))
# extract and transform
ls_df_t = []
dt_errors = {}
for date_str in ls_date_str:
    print(date_str)
    df_sym = get_df_sym(db, date_str)
    for i, tup in tqdm(df_sym.iterrows(), total=df_sym.shape[0]):
        sym = tup['sym']
        try:
            df_c = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
            ls_df_t.append(df_c[df_c['divergence']!=''])
        except Exception as e:
            dt_errors[sym] = ERROR_EXCEPTION.format(type(e).__name__, e)
if dt_errors:
    num_runs = df_sym.shape[0]*len(ls_date_str)
    [print(ERROR_SUMMARY.format(sym, dt_errors[sym])) for sym in dt_errors]
    print(ERROR_PCT.format(len(dt_errors), num_runs, len(dt_errors)/num_runs))

# save df_train
if ls_df_t:
    df_t = pd.concat(ls_df_t)
ls_col = [
    'is_profit',
    'rsi14',
    'sma9_var',
    'sma180_var',
    'vwap_var',
    'spread14_e',
    'volume14_34_var',
    'prev_close_var',
    'prev_floor_var',
    'prev_ceil_var',
    'prev1_candle_score',
    'prev2_candle_score',
    'prev3_candle_score',
    'mins_from_start',
    'valley_interval_mins',
    'valley_close_score',
    'valley_rsi_score',
    'day_open_var',
    'open_from_prev_close_var',
    'ceil_var',
    'floor_var',
    ###
    'sym',
    'datetime',
    'profit',
    'prev_close',
    'divergence',
]
df_t = df_t[ls_col]
df_t = df_t.dropna()
#cat_cols = df_t.select_dtypes(include=object).columns
#df_t[cat_cols] = df_t[cat_cols].fillna('none').astype('category')
print(df_t.info())

# df_train - Export
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
f = os.path.join(DIR_TRAIN, f'df_train_{timestamp}.parquet')
df_t.to_parquet(f)
print(MSG_SAVED.format(f))
beeps()