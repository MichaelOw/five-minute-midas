import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_google_link(text):
    '''Return link to google news of the input text
    Args:
        text (str)
    Returns:
        link (str)
    '''
    link = f'https://www.google.com/search?q={text}+stock&tbm=nws'
    return link

def get_yahoo_link(text):
    '''Return link to yahoo stock chart of the input text
    Args:
        text (str)
    Returns:
        link (str)
    '''
    link = f'https://sg.finance.yahoo.com/chart/{text}'
    return link

def get_df_parquet(ls_f, directory=''):
    '''Loads parquet files indicated in ls_f and returns a consolidated DataFrame
    Args:
        ls_f (List of str)
        directory (str)
    Returns:
        df (pandas.DataFrame)
    '''
    ls_df = []
    ls_date_str = []
    for f in ls_f:
        df = pd.read_parquet(os.path.join(directory, f))
        ls_date_str_new = list(df.datetime.dt.date.astype(str).unique())
        for date_str in ls_date_str_new:
            assert date_str not in ls_date_str, f'"{date_str}" is found more than once!'
            ls_date_str.append(date_str)
        ls_df.append(df)
    df = pd.concat(ls_df)
    return df

def get_df_sym(db, date_str=''):
    '''Returns df_sym
    Args:
        db (DataBase object)
        date_str (str) - If None then will return based on latest date
    Returns:
        df_sym (pd.DataFrame)
            sym (str)
            sec (str)
            ind (str)
            long_name (str)
    '''
    
    if len(date_str) == 10:
        q_modifier = '''WHERE DATE(datetime) = '{}' '''.format(date_str)
    else:
        q_modifier = '''WHERE DATE(datetime) = (SELECT MAX(DATE(datetime)) FROM prices_m)'''
    q = '''
        SELECT DISTINCT prices_m.sym, sec, ind, long_name
          FROM prices_m
          LEFT JOIN stocks
            ON stocks.sym=prices_m.sym
               {}
    '''.format(q_modifier)
    df_sym = pd.read_sql(q, db.conn)
    return df_sym

def append_to(f, string):
    '''Appends string to file
    Args:
        f (str): Filename
        string (str): String to append to file
    '''
    with open(f, 'a') as txt_file:
        txt_file.write(f'\n{string}')

def timer_dec(func):
    '''Decorator which prints how long a function took to run
    Args:
        func (Function)
    '''
    def timed_func(*args, **kwargs):
        print(f'Function ({func.__name__}) started... ')
        time_start = time.time()
        func(*args, **kwargs)
        dur_min = str((time.time()-time_start)/60)[:3]
        print(f'Function ({func.__name__}) completed in {dur_min} mins... ')
    return timed_func

def plot_divergences(df, title, save_dir = ''):
    '''Plot the peaks and valleys of [close] value against [datetime]
    Args:
        df (pandas.Dataframe): datetime, close, sma180, rsi14, vwap, peak_valley, divergence, profit_proba
        title (str): Chart title
        save_dir (str): Directory to save image e.g. 'D:\\Program Files (x86)\\Stocks', no saving if empty
    '''
    # new columns
    if 'profit_proba' not in df.columns: df['profit_proba'] = -1
    df['period'] = df['Date'] if 'Date' in df.columns else df['datetime'].dt.time.astype('str').str[:5]
    df['close_div'] = np.where(df['divergence']!='', df['close'], np.nan)
    df['close_div_weak'] = np.where((df['divergence']!='')&(df['profit_proba']>0.5), df['close'], np.nan)
    df['close_div_strong'] = np.where((df['divergence']!='')&(df['profit_proba']>0.75), df['close'], np.nan)
    # setup plot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
    # top plot - price line plot
    sns.lineplot(data=df, x='period', y='close', ax=axs[0])
    sns.lineplot(data=df, x='period', y='sma180', color='y', ax=axs[0])
    sns.lineplot(data=df, x='period', y='vwap', color='r', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div', color='r', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div_weak', color='orange', ax=axs[0])
    sns.scatterplot(data=df, x='period', y='close_div_strong', color='lime', ax=axs[0])
    # top plot - profit proba labels
    for _, point in df.iterrows():
        x = point['period']
        y = point['close']
        value = point['profit_proba']
        value2 = point['profit']
        div = point['divergence']
        if div and value:
            axs[0].text(x, y*0.992, f'{round(value, 2)}\n({round(value2, 2)})', fontsize=9)
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
    # title
    fig.suptitle(title)
    # save image if directory provided
    if save_dir:
        profit_proba_max = round(df['profit_proba'].max(), 2)
        time_str = datetime.datetime.now().strftime('%H%M')
        plt.savefig(f'{save_dir}\\{sym}_{time_str}_{profit_proba_max}.png', bbox_inches='tight')
    plt.show()

def db_remove_dups_stocks(db):
    '''Removes duplicates in table prices_d
    Args:
        db (DataBase object)
    '''
    print('Removing duplicates...')
    q = '''
        DELETE
          FROM stocks
         WHERE ROWID NOT IN
               (SELECT MAX(ROWID)
                  FROM stocks
                 GROUP BY sym)
    '''
    db.execute(q)
    q = '''
        DELETE
          FROM stocks_error
         WHERE ROWID NOT IN
               (SELECT MAX(ROWID)
                  FROM stocks_error
                 GROUP BY sym)
    '''
    db.execute(q)

def db_remove_dups_prices_m(db, date_str):
    '''Removes duplicates in table prices_m.
    Starts looking in dates >= date_str
    Args:
        db (DataBase object)
        date_str (str): e.g. 2020-12-24
    '''
    print('Removing duplicates...')
    q = '''
        DELETE
          FROM prices_m
         WHERE DATE(datetime)>='{}'
           AND ROWID NOT IN
               (SELECT MAX(ROWID)
                 FROM prices_m
                WHERE DATE(datetime)>='{}'
                GROUP BY sym, datetime)
    '''.format(date_str, date_str)
    db.execute(q)

def db_remove_dups_prices_d(db):
    '''Removes duplicates in table prices_d.
    Starts looking in dates >= date_str
    Args:
        db (DataBase object)
    '''
    print('Removing duplicates...')
    q = '''
        DELETE
          FROM prices_d
         WHERE ROWID NOT IN
               (SELECT MAX(ROWID)
                 FROM prices_d
                GROUP BY sym, date)
    '''
    db.execute(q)