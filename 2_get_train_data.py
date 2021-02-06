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
from configparser import ConfigParser
from src.utils_beeps import beeps
from src.utils_stocks import get_df_c
from src.utils_general import get_df_sym
from src.utils_general import plot_divergences
from src.utils_date import get_ls_date_str_from_db
# directories
DIR_DB = os.path.join(os.getcwd(), 'data', 'db')
DIR_TRAIN = os.path.join(os.getcwd(), 'data', 'train')
F_CFG = os.path.join(os.getcwd(), 'config.ini')
# objects
db = DataBase([], DIR_DB)
cfg = ConfigParser()
cfg.read(F_CFG)
# constants
ERROR_EXCEPTION = 'Error: Exception found ({}: {})'
ERROR_SUMMARY = '{} - {}'
ERROR_PCT = 'Errors: {}/{} {:.3f}'
MSG_DATE_RANGE = 'Creating df_train for date range: {} to {}'
MSG_SAVED = '{} saved!'
FILENAME_TRAIN = 'df_train_{}.parquet'
CFG_SECTION = 'get_train_data'
LIVE_DATA = cfg.getint(CFG_SECTION, 'LIVE_DATA')
TARGET_PROFIT = cfg.getfloat(CFG_SECTION, 'TARGET_PROFIT')
TARGET_LOSS = cfg.getfloat(CFG_SECTION, 'TARGET_LOSS')
DATE_START = cfg.get(CFG_SECTION, 'DATE_START')
DATE_END = cfg.get(CFG_SECTION, 'DATE_END')
LS_DATE_STR = get_ls_date_str_from_db(DATE_START, DATE_END, db)
LS_COL = [
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
    'day_sma9_var',
    'day_sma180_var',
    'day_rsi14',
    ###
    'sym',
    'datetime',
    'profit',
    'prev_close',
    'divergence',
]
# extract and transform
print(MSG_DATE_RANGE.format(LS_DATE_STR[0], LS_DATE_STR[-1]))
ls_df_t = []
for date_str in LS_DATE_STR:
    dt_errors = {}
    print(date_str)
    df_sym = get_df_sym(db, date_str)
    for i, tup in tqdm(df_sym.iterrows(), total=df_sym.shape[0]):
        sym = tup['sym']
        try:
            df_c = get_df_c(sym, date_str, LIVE_DATA, db, TARGET_PROFIT, TARGET_LOSS)
            ls_df_t.append(df_c[df_c['divergence']!=''][ls_col])
        except Exception as e:
            dt_errors[sym] = ERROR_EXCEPTION.format(type(e).__name__, e)
    if dt_errors:
        num_runs = df_sym.shape[0]*len(LS_DATE_STR)
        [print(ERROR_SUMMARY.format(sym, dt_errors[sym])) for sym in dt_errors]
        print(ERROR_PCT.format(len(dt_errors), num_runs, len(dt_errors)/num_runs))
if ls_df_t:
    # save df_train
    df_t = pd.concat(ls_df_t)
    df_t = df_t.dropna()
    print(df_t.info())
    # df_train - Export
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    f = os.path.join(DIR_TRAIN, FILENAME_TRAIN.format(timestamp))
    df_t.to_parquet(f)
print(MSG_SAVED.format(f))
beeps()