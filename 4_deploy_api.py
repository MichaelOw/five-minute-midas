import time
import json
import pickle
import datetime
import traceback
import numpy as np
import pandas as pd
from src.db import DataBase
from src.utils_general import beeps
from src.utils_general import timer_dec
from src.utils_stocks import get_df_c
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask #request, redirect, url_for, flash, jsonify
dir_db = json.load(open('dir.txt'))['dir_db']
dir_models = json.load(open('dir.txt'))['dir_models']

# user parameters
refresh_mins = 400
date_str = '2020-12-11'
live_data = 1
f_model = 'tup_model_2020-12-06_1640.p'

# load model
print('Loading...', end = '')
tup_model = pickle.load(open(dir_models+f_model, 'rb'))

# api
app = Flask(__name__)
print('Done!')

@app.route('/proba', methods=['POST'])
def api_get_df_proba_sm():
    db = DataBase([], dir_db)
    q = '''
    SELECT *
      FROM proba
     WHERE sym + datetime_update IN
           (SELECT sym + MAX(datetime_update)
              FROM proba
             GROUP BY sym)
    '''
    df_proba = pd.read_sql(q, db.conn)
    j_df_proba = df_proba.to_json(orient='split')
    return j_df_proba

def get_df_sym_filter(db):
    ls_sec = [       
        'Technology',
        'Utilities',
        'Communication Services',
        #'Consumer Defensive',
        #'Consumer Cyclical',
        #'Energy',
        #'Basic Materials',
        #'Real Estate',
        #'Industrials,'
        #'Financial',
        #'Healthcare',
        #'Financial Services',
    ]
    ls_ind = [
        'Auto Manufacturers',
        'Internet Retail',
        'Education & Training Services',
        'Packaged Foods',
        'Grocery Stores',
    ]
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

def get_df_proba(df_c, tup_model):
    '''Get df_proba
        Args:
        df_c (pandas.DataFrame)
        tup_model (tup):
            q (str): Pandas query string
            ls_col (List of str)
            full_pipe (sklearn.pipeline.Pipeline)
    Returns:
        df_proba (pandas.DataFrame)
    '''
    q, ls_col, full_pipe = tup_model
    # Remove outliers and non-relevant data 
    df = df_c.query(q).copy()
    if df.empty: return pd.DataFrame()
    s_sym = df['sym']
    s_datetime = df['datetime']
    s_timestamp = [datetime.datetime.now()]*df.shape[0]
    df = df[ls_col]
    arr_proba = full_pipe.predict_proba(df)
    df_proba = pd.DataFrame({
        'sym':s_sym,
        'datetime':s_datetime,
        'my_index':list(df.index),
        'proba':arr_proba[:,1],
        'datetime_update':s_timestamp,
    })
    return df_proba

@timer_dec
def update_stuff():
    global dir_db
    global tup_model
    global j_df_proba
    global date_str
    global live_data
    db = DataBase([], dir_db)
    ls_df_proba = []
    target_profit = 0.011
    target_loss = -0.031
    df_sym = get_df_sym_filter(db)
    for i, tup in df_sym.iloc[:].iterrows():
        if i%100==0: print(i, df_sym.shape[0])
        sym = tup['sym']
        try:
            df_c = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
            df_proba = get_df_proba(df_c, tup_model)
            if not df_proba.empty: df_proba.to_sql('proba', db.conn, if_exists='append', index=0)
        except Exception as e:
            print(sym, type(e).__name__, e.args) #traceback.print_exc()

# apscheduler
sched = BackgroundScheduler(daemon=True)
sched.add_job(update_stuff)
sched.add_job(update_stuff, 'interval', minutes=refresh_mins)
sched.start()
if __name__ == '__main__':
    db = DataBase([], dir_db)
    db.execute('DELETE FROM proba')
    app.run(debug=False, host='0.0.0.0')