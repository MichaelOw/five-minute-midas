import os
import time
import json
import pickle
import datetime
import traceback
import threading
import numpy as np
import pandas as pd
from src.db import DataBase
from flask import Flask, request
from src.utils_stocks import get_df_c
from src.utils_general import timer_dec
dir_db = os.path.join(os.getcwd(), 'data', 'db')
dir_models = os.path.join(os.getcwd(), 'data', 'models')

# user parameters
buffer_seconds = 100000
date_str = '2020-12-17'
live_data = 0
f_model = 'tup_model_2020-12-06_1640.p'
sym_limit = 100 #None

# load model
print('Loading...', end = '')
with open(os.path.join(dir_models, f_model), 'rb') as f:
    tup_model = pickle.load(f)

# api
app = Flask(__name__)
print('Done!')

@app.route('/df_proba', methods=['POST'])
def api_get_df_proba():
    '''API that returns all predictions in dataframe in JSON'''
    global dir_db
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
    db.close()
    return j_df_proba

@app.route('/df_proba_sm', methods=['POST'])
def api_get_df_proba_sm():
    '''API that returns prediction summary in dataframe in JSON'''
    global dir_db
    db = DataBase([], dir_db)
    q = '''
    SELECT *
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
    '''API that returns full price dataframe in JSON for input symbol(s)'''
    global dir_db
    global tup_model
    global date_str
    global live_data
    db = DataBase([], dir_db)
    j_data = request.get_json()
    ls_sym = json.loads(j_data)['ls_sym']
    time_str = json.loads(j_data)['time_str']
    target_profit = 0.011
    target_loss = -0.031
    try:
        ls_df = []
        for sym in ls_sym:
            df = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
            df = df[df['datetime'].dt.strftime('%H%M')<=time_str]
            df_proba = get_df_proba(df, tup_model)
            if not df_proba.empty:
                df = pd.merge(df, df_proba[['sym','datetime','proba']], how='left', on=['sym', 'datetime'])
            else:
                df['proba'] = None
            ls_df.append(df)
        df_c = pd.concat(ls_df)
        j_df_c = df_c.to_json(orient='split')
        db.close()
        return j_df_c
    except Exception as e:
        print(sym, type(e).__name__, e.args) #traceback.print_exc()
        db.close()
        return pd.DataFrame().to_json(orient='split')

def get_df_sym_filter(db):
    '''Returns dataframe of stock symbols in
    pre-selected sectors and industries
    Args:
        db (DataBase)
    '''
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
    '''Returns dataframe with only profit predictions
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
def update_predictions():
    '''Runs an iteration of model predictions on
    selected symbols and saves output in database
    '''
    global dir_db
    global tup_model
    global j_df_proba
    global date_str
    global live_data
    global sym_limit
    global buffer_seconds
    db = DataBase([], dir_db)
    ls_df_proba = []
    target_profit = 0.011
    target_loss = -0.031
    df_sym = get_df_sym_filter(db)
    df_sym = df_sym.iloc[:sym_limit]
    while 1:
        for i, tup in df_sym.iterrows():
            if i%100==0: print(i, df_sym.shape[0])
            sym = tup['sym']
            try:
                df_c = get_df_c(sym, date_str, live_data, db, target_profit, target_loss)
                df_proba = get_df_proba(df_c, tup_model)
                if not df_proba.empty: df_proba.to_sql('proba', db.conn, if_exists='append', index=0)
            except Exception as e:
                print(sym, type(e).__name__, e.args) #traceback.print_exc()
        print(f'Update complete, waiting for {buffer_seconds} seconds till next update...')
        time.sleep(buffer_seconds)

if __name__ == '__main__':
    db = DataBase([], dir_db)
    db.execute('DELETE FROM proba')
    x = threading.Thread(target=update_predictions, daemon=True)
    x.start()
    app.run(debug=False, host='0.0.0.0')