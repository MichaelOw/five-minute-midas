import os
import json
import pickle
import winsound
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from tqdm import tqdm
from src.db import DataBase
import matplotlib.pyplot as plt
from src.utils_stocks import get_df_c
from src.utils_general import get_df_sym
from src.utils_general import plot_divergences
from src.utils_date import get_ls_date_str_from_db
dir_db = os.path.join(os.getcwd(), 'data', 'db')
dir_train = os.path.join(os.getcwd(), 'data', 'train')
db = DataBase([], dir_db=dir_db)
def beeps(n=3):
    '''Produce n beeps. Default 3'''
    for _ in range(n):
        winsound.Beep(200,500)
# params
live_data = 0
target_profit = 0.011
target_loss = -0.031
ls_date_str = get_ls_date_str_from_db('2020-12-14', '2022-01-01', db) #'2020-06-23'
print('Creating df_train for date range: {} to {}'.format(ls_date_str[0], ls_date_str[-1]))
# extract and transform
count, count_e = 0, 0
ls_df_t = []
dt_errors = {}
for date_str in ls_date_str:
    print(date_str)
    df_sym = get_df_sym(db, date_str)
    for i, tup in tqdm(df_sym.iterrows(), total=df_sym.shape[0]):
        count+=1
        try:
            df_c = get_df_c(tup['sym'], date_str, live_data, db, target_profit, target_loss)
            ls_df_t.append(df_c[df_c['divergence']!=''])
        except Exception as e:
            dt_errors[tup['sym']] = f'Error: {type(e).__name__} {e}'
            count_e+=1
print('Errors:', count_e, count, round(count_e/count, 3))
print(dt_errors)
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
f = os.path.join(dir_train, f'df_train_{timestamp}.parquet')
df_t.to_parquet(f)
print(f, 'saved!')
beeps()