import os
import time
import json
import pickle
import datetime
import traceback
import threading
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytz import timezone
from src.db import DataBase
from flask import Flask, request
from configparser import ConfigParser
from src.utils_stocks import get_df_c
from src.utils_stocks import check_prices_d_updated
from src.utils_general import timer_dec
from src.utils_model import get_df_proba
# directories
DIR_DB = os.path.join(os.getcwd(), 'data', 'db')
DIR_MODELS = os.path.join(os.getcwd(), 'data', 'models')
F_CFG = os.path.join(os.getcwd(), 'config.ini')
# objects
cfg = ConfigParser()
cfg.read(F_CFG)
# constants
DATE_STR_TDY = (datetime.datetime.now()
                    .astimezone(timezone('America/New_York'))
                    .strftime('%Y-%m-%d'))
ERROR_EXCEPTION = 'Error: Exception found ({}: {})'
ERROR_EXCEPTION_SYM = 'Error: Exception found for {} ({}: {})'
ERROR_SUMMARY = '{} - {}'
ERROR_PCT = 'Errors: {}/{} {:.3f}'
MSG_SKIP = 'Skipping these symbols: {}'
MSG_DEFAULT_DATE = 'No date entered, using today date: {}'
CFG_SECTION = 'deploy_api'
F_MODEL = cfg.get(CFG_SECTION, 'F_MODEL')
BUFFER_SECONDS = cfg.getfloat(CFG_SECTION, 'BUFFER_SECONDS')
LIVE_DATA = cfg.getint(CFG_SECTION, 'LIVE_DATA')
DATE_STR = cfg.get(CFG_SECTION, 'DATE_STR')
TARGET_PROFIT = cfg.getfloat(CFG_SECTION, 'TARGET_PROFIT')
TARGET_LOSS = cfg.getfloat(CFG_SECTION, 'TARGET_LOSS')
ERROR_THRESHOLD = cfg.getint(CFG_SECTION, 'ERROR_THRESHOLD')
LS_SEC = json.loads(cfg.get(CFG_SECTION,'LS_SEC'))
LS_IND = json.loads(cfg.get(CFG_SECTION,'LS_IND'))
pause = 0
# get date
if not DATE_STR:
    print(MSG_DEFAULT_DATE.format(DATE_STR_TDY))
    DATE_STR = DATE_STR_TDY
# init database
db = DataBase([], DIR_DB)
db.execute('DELETE FROM proba')
check_prices_d_updated(DATE_STR_TDY, db)
# load model
with open(os.path.join(DIR_MODELS, F_MODEL), 'rb') as f:
    tup_model = pickle.load(f)
# api
app = Flask(__name__)
# functions
@app.route('/df_proba_sm', methods=['POST'])
def api_get_df_proba_sm():
    '''API that returns prediction summary in dataframe in JSON
    Returns:
        j_df_proba_sm (JSON of pandas.Dataframe)
    '''
    db = DataBase([], DIR_DB)
    q = '''
    SELECT sym, datetime, my_index, proba, datetime_update
      FROM proba
     WHERE sym + datetime_update IN
           (SELECT sym + MAX(datetime_update)
              FROM proba
             GROUP BY sym)
    '''
    df = pd.read_sql(q, db.conn)
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
    j_df_proba_sm = df_proba_sm.to_json(orient='split')
    db.close()
    return j_df_proba_sm

@app.route('/df_c', methods=['POST'])
def api_get_df_c():
    '''API that returns full cooked price dataframe in JSON for input symbol(s)
    Returns:
        j_df_c (JSON of pandas.Dataframe)
    '''
    global pause
    pause = 1
    time.sleep(2)
    db = DataBase([], DIR_DB)
    j_data = request.get_json()
    ls_sym = json.loads(j_data)['ls_sym']
    time_str = json.loads(j_data)['time_str']
    try:
        ls_df = []
        for sym in ls_sym:
            time.sleep(BUFFER_SECONDS)
            df = get_df_c(sym, DATE_STR, LIVE_DATA, db, TARGET_PROFIT, TARGET_LOSS)
            df = df[df['datetime'].dt.strftime('%H%M')<=time_str]
            df_proba = get_df_proba(df, tup_model)
            if not df_proba.empty:
                df = pd.merge(df
                    ,df_proba[['sym','datetime','proba']]
                    ,how='left'
                    ,on=['sym', 'datetime'])
            else:
                df['proba'] = None
            ls_df.append(df)
        df_c = pd.concat(ls_df)
        j_df_c = df_c.to_json(orient='split')
    except Exception as e:
        print(ERROR_EXCEPTION_SYM.format(sym, type(e).__name__, e.args))
        j_df_c = pd.DataFrame().to_json(orient='split')
    db.close()
    pause = 0
    return j_df_c

def get_df_sym_filter(db, ls_sec, ls_ind):
    '''Returns dataframe of stock symbols in
    pre-selected sectors and industries
    Args:
        db (DataBase)
    '''
    q = '''
        SELECT stocks.sym
               ,stocks.long_name
               ,stocks.sec
               ,stocks.ind
               ,stocks.quote_type
               ,stocks.fund_family
               ,stocks.summary
          FROM prices_d
          LEFT JOIN stocks
            ON stocks.sym = prices_d.sym
         WHERE DATE(date) = (SELECT MAX(DATE(date)) FROM prices_d WHERE sym='IBM')
           AND adj_close > 5
           AND volume > 100000
    '''
    df = pd.read_sql(q, db.conn)
    index = (df['sec'].isin(ls_sec))|(df['ind'].isin(ls_ind))
    df = df[index]
    df = df.reset_index()
    return df

@timer_dec
def update_predictions():
    '''Runs an iteration of model predictions on
    selected symbols and saves output in database
    '''
    db = DataBase([], DIR_DB)
    df_sym = get_df_sym_filter(db, LS_SEC, LS_IND)
    c_error = collections.Counter()
    ls_skip = []
    while 1:
        dt_error = {}
        for i, tup in tqdm(df_sym.iterrows(), total=df_sym.shape[0]):
            while pause:
                time.sleep(BUFFER_SECONDS)
            sym = tup['sym']
            if sym not in ls_skip:
                try:
                    time.sleep(BUFFER_SECONDS)
                    df_c = get_df_c(sym, DATE_STR, LIVE_DATA, db, TARGET_PROFIT, TARGET_LOSS)
                    df_proba = get_df_proba(df_c, tup_model)
                    if not df_proba.empty:
                        df_proba.to_sql('proba', db.conn, if_exists='append', index=0)
                except Exception as e:
                    dt_error[sym] = ERROR_EXCEPTION.format(type(e).__name__, e) # traceback.print_exc()
                    c_error.update([sym])
        if dt_error:
            num_runs = df_sym.shape[0]
            [print(ERROR_SUMMARY.format(sym, dt_error[sym])) for sym in dt_error]
            print(ERROR_PCT.format(len(dt_error), num_runs, len(dt_error)/num_runs))
        ls_skip =  [k for k, v in c_error.items() if v > ERROR_THRESHOLD] # skip symbols with too many errors
        print(MSG_SKIP.format(ls_skip))

if __name__ == '__main__':
    x = threading.Thread(target=update_predictions, daemon=True)
    x.start()
    app.run(debug=False, host='0.0.0.0')
    x.join()