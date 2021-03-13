import os
import sys
import datetime
import numpy as np
import scipy.signal
import pandas as pd
import yfinance as yf
from contextlib import contextmanager
from src.utils_date import add_days
from src.utils_date import prev_weekday
#from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

ERROR_NO_MINUTE_DATA_YTD = 'Skip: Missing minute-level data for yesterday'
ERROR_NO_MINUTE_DATA_TDY = 'Skip: Missing minute-level data for today'
ERROR_CANDLES_PER_DAY = 'Skip: Insufficient candles today ({} less than {})'
ERROR_NULL_COL = 'Skip: NULL value in df_i columns ({})'
ERROR_NULL_DAY_LEVEL_IND = 'Skip: NULL value in day-level indicators'
ERROR_PRICES_D_NOT_UPDATE = 'Error: prices_d not updated, latest date found: {}'

@contextmanager
def suppress_stdout():
    '''Decorator to supress function output to sys.stdout'''
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_ls_sym():
    '''Returns list of tickers from nasdaqtrader.com
        Duplicates and strings with length > 5 are removed
    Returns:
        ls_sym (List of str)
    '''
    #df_symbols = get_nasdaq_symbols()
    #ls_sym = df_symbols.index.to_list()
    ls_urls = [
        'http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
        ,'http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
    ]
    ls_sym = []
    for i, url in enumerate(ls_urls):
        df = pd.read_csv(url, sep='|')
        for col in list(df):
            if col in ['ACT Symbol', 'Symbol']: df['sym'] = df[col]
        ls_sym+=df[df['sym'].str.len()<=5]['sym'].to_list()
    ls_sym = list(set(ls_sym)) # remove duplicates
    return ls_sym

def get_df_prices(sym, start_str, end_str):
    '''Return dataframe with minute-level stock price data
    from start date to end date (inclusive).
    Args:
        sym (str): Ticker symbol e.g. 'BYND'
        start_str (str): Start date string e.g. '2020-07-18'
        end_str (str): End date string e.g. '2020-07-18'
    Returns:
        df (pandas.Dataframe)
    '''
    assert start_str <= end_str
    end_str_mod=add_days(end_str, 3)
    with suppress_stdout():
        df = yf.download(sym,
            start=start_str,
            end=end_str_mod,
            interval='1m',
            progress=0,
            prepost=True).reset_index()
    is_date_range = ((df['Datetime'].dt.date.astype('str')>=start_str)
                     &(df['Datetime'].dt.date.astype('str')<=end_str))
    df = df[is_date_range]
    df['Datetime'] = df['Datetime'].dt.tz_localize(None) #remove timezone
    is_reg_hours = ((df['Datetime'].dt.time.astype('str')>='09:30:00')
                    &(df['Datetime'].dt.time.astype('str')<='15:59:00'))
    df['is_reg_hours'] = np.where(is_reg_hours, 1, 0)
    df['sym'] = sym
    df = df.rename(columns={
        'Datetime':'datetime',
        'Open':'open',
        'High':'high',
        'Low':'low',
        'Adj Close':'adj_close',
        'Volume':'volume'
    })
    ls_col = [
        'sym',
        'datetime',
        'open',
        'high',
        'low',
        'adj_close',
        'volume',
        'is_reg_hours',
    ]
    return df[ls_col]

def add_rsi(df, rsi_period):
    '''Returns dataframe with additional columns:
        rsi (float)
    Args:
        df (pandas.DataFrame): Must be index sorted by datetime:
            adj_close (float)
        rsi_period (int): Number of rsi periods
    Returns:
        df (pandas.DataFrame)
    '''
    chg = df['adj_close'].diff(1)
    gain = chg.mask(chg<0,0)
    loss = chg.mask(chg>0,0)
    avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
    rs = abs(avg_gain/avg_loss)
    rsi = 100 - (100/(1+rs))
    df['rsi14'] = rsi
    return df

def add_vwap(df):
    '''Returns dataframe with additional columns:
        vwap (float): Volume Weighted Average Price
        vwap_var (float): % variance of close from vwap
    Args:
        df (pandas.DataFrame): Dataframe with at least columns:
            datetime
            open
            high
            low
            adj_close
            volume
    Returns:
        df (pandas.DataFrame)
    '''
    df['vwap'] = (df['volume']*(df['high']+df['low']+df['adj_close'])/3).cumsum()/df['volume'].cumsum()
    df['vwap'] = df['vwap'].fillna(df['adj_close'])
    df['vwap_var'] = (df['adj_close']/df['vwap'])-1
    return df

def get_df_i(sym, date_str, live_data, db, num_candles_min = 200):
    '''Returns interim dataframe with price data and
    trading indicators for input symbol and date
    Args:
        sym (str)
        date_str  (str)
        live_data (int)
        db (Database object)
        num_candles_min (int)
    Returns:
        df_i (pandas.Dataframe)
    '''
    start_str = prev_weekday(date_str) #start 1 day early to get prev day data for rsi etc
    end_str = add_days(date_str, 3) #extend end date string due to bug
    if live_data:
        with suppress_stdout():
            df = yf.download(sym,
                start=start_str,
                end=end_str,
                interval='1m',
                prepost = False,
                progress=0).reset_index()
        df['Datetime'] = df['Datetime'].dt.tz_localize(None) #remove timezone
        df = df.rename(columns={'Adj Close':'adj_close',
                                   'Datetime':'datetime',
                                   'Open':'open',
                                   'High':'high',
                                   'Low':'low',
                                   'Volume':'volume'})
    else:
        q = '''
            SELECT *
              FROM prices_m
             WHERE is_reg_hours = 1
               AND sym='{}'
               AND DATE(datetime)>='{}'
               AND DATE(datetime)<='{}'
             ORDER BY datetime
        '''.format(sym, start_str, date_str)
        df = pd.read_sql(q, db.conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
    df['date_str'] = df['datetime'].dt.date.astype('str')
    if df[df['date_str']==start_str].empty:
        raise Exception(ERROR_NO_MINUTE_DATA_YTD)
    if df[df['date_str']==date_str].empty:
        raise Exception(ERROR_NO_MINUTE_DATA_TDY)
    num_candles_today = df[df['date_str']==date_str].shape[0]
    if num_candles_today<num_candles_min and not live_data:
        raise Exception(ERROR_CANDLES_PER_DAY.format(num_candles_today, num_candles_min))
    df = df[df['date_str']<=date_str]
    df = df[df['date_str']>=start_str]
    df['sma9'] = df['adj_close'].rolling(9).mean()
    df['sma90'] = df['adj_close'].rolling(90).mean()
    df['sma180'] = df['adj_close'].rolling(180).mean()
    df['sma180'] = df['sma180'].fillna(df['sma90'])
    df['sma9_var'] = (df['adj_close']/df['sma9'])-1
    df['sma180_var'] = (df['adj_close']/df['sma180'])-1
    df = add_rsi(df, 14)
    df['spread']=((df['adj_close']/df['open'])-1).abs()
    df['spread14_e']=df['spread'].ewm(span=14).mean()
    df['volume14'] = df['volume'].rolling(14).mean()
    df['volume34'] = df['volume'].rolling(34).mean()
    df['volume14_34_var'] = (df['volume14']/df['volume34'])-1
    df['volume14_34_var'] = df['volume14_34_var'].fillna(0.0)
    prev_close = df[df['date_str']==start_str]['adj_close'].to_list()[-1]
    prev_floor = df[df['date_str']==start_str]['adj_close'].min()
    prev_ceil = df[df['date_str']==start_str]['adj_close'].max()
    df['prev_close'] = prev_close
    df['prev_close_var'] = df['adj_close']/prev_close - 1
    df['prev_floor_var'] = (df['adj_close']/prev_floor)-1
    df['prev_ceil_var'] = (df['adj_close']/prev_ceil)-1
    df['candle_score'] = df['adj_close']/df['open']-1
    df['prev1_candle_score'] = df['candle_score'].shift(1)
    df['prev2_candle_score'] = df['candle_score'].shift(2)
    df['prev3_candle_score'] = df['candle_score'].shift(3)
    df = df[df['date_str']==date_str]
    df = add_vwap(df)
    df = df.rename(columns={'adj_close':'close'})
    ls_col = [
        'datetime', 
        'close', 
        'sma9',
        'sma180',
        'rsi14', 
        'vwap', 
        'sma9_var',
        'sma180_var',
        'vwap_var',
        'spread14_e',
        'volume14_34_var',
        'prev_close',
        'prev_close_var',
        'prev_floor_var',
        'prev_ceil_var',
        'prev1_candle_score',
        'prev2_candle_score',
        'prev3_candle_score',   
    ]
    df = df[ls_col]
    ls_col_na = df.columns[df.isna().any()].tolist()
    if ls_col_na:
        raise Exception(ERROR_NULL_COL.format(ls_col_na))
    return df.reset_index(drop=1)

def add_peaks_valleys(df, order=5):
    '''Returns Dataframe with additional columns:
        peak_valley - 1 if peak, -1 if valley, 0 o.w.
    Args:
        df (pandas.DataFrame): Dataframe with at least columns:
            datetime
            close
        order (int): How many points on each side to use for the comparison to consider
    Returns:
        df (pandas.DataFrame)
    '''
    df['peak_valley'] = 0
    col_peak_valley = list(df).index('peak_valley')
    peak_indexes = scipy.signal.argrelextrema(np.array(df['close']), np.greater, order = order)[0]
    valley_indexes = scipy.signal.argrelextrema(np.array(df['close']), np.less, order = order)[0]
    df.iloc[peak_indexes, col_peak_valley] = 1
    df.iloc[valley_indexes, col_peak_valley] = -1
    return df

def add_valley_variances(df):
    '''Returns Dataframe with additional columns:
        valley_close_pct_chg (float): % change in close of current and previous valley e.g. 1% -> 0.01
        valley_rsi_diff (float): Change in rsi of current and previous valley
        valley_interval_mins (float): Minutes since last valley
    Args:
        df (pandas.DataFrame): Dataframe with at least columns:
            datetime
            close
            rsi14
            peak_valley
    Returns:
        df (pandas.DataFrame)
    '''
    df['valley_close'] = np.where(df['peak_valley']==-1, df['close'], np.nan)
    df['valley_rsi'] = np.where(df['peak_valley']==-1, df['rsi14'], np.nan)
    df['valley_datetime'] = pd.to_datetime(np.where(df['peak_valley']==-1, df['datetime'], pd.NaT))
    df['valley_close'] = df['valley_close'].ffill()
    df['valley_rsi'] = df['valley_rsi'].ffill()
    df['valley_datetime'] = df['valley_datetime'].ffill()
    df['valley_close_pct_chg'] = df['valley_close'].pct_change()
    df['valley_rsi_diff'] = df['valley_rsi'].diff()
    df['valley_interval_mins'] = df['valley_datetime'].diff().astype('timedelta64[m]')
    df = df.drop(columns=['valley_close'
                            ,'valley_rsi'
                            ,'valley_datetime'])
    return df

def add_divergences(df, close_buffer=0, rsi_buffer=0):
    '''Returns Dataframe with additional columns:
        divergence (str):
            'bull_reg' - Regular bullish divergence i.e. Lower price valleys, but rise in RSI
            'bull_hid' - Hidden bullish divergence i.e. Higher price valleys, but drop in RSI
            '' - No divergence
    Args:
        df (pandas.DataFrame): Dataframe with at least columns:
            datetime
            valley_close_pct_chg
            valley_rsi_diff
        close_buffer (float): Price change must be at least this % change to count as divergence, e.g 1.5 -> 1.5% 
        rsi_buffer (float): RSI change must be at least this change to count as divergence
    Returns:
        df (pandas.DataFrame)
    '''
    df['divergence'] = ''
    df['divergence'] = np.where((df['valley_close_pct_chg'] < -(close_buffer/100))
                                    &(df['valley_rsi_diff'] > rsi_buffer)
                                ,'bull_reg'
                                ,df['divergence'])
    df['divergence'] = np.where((df['valley_close_pct_chg'] > (close_buffer/100))
                                    &(df['valley_rsi_diff'] < -rsi_buffer)
                                ,'bull_hid'
                                ,df['divergence'])
    return df

def add_additional_measures(df, sym):
    '''Add last few features to Dataframe
    Args:
        df (pandas.Dataframe)
    Returns:
        df (pandas.Dataframe)
    '''
    df['mins_from_start'] = (df['datetime']-df['datetime'].min()).astype('timedelta64[m]')
    df['valley_close_score'] = df['valley_close_pct_chg'].abs()*100
    df['valley_rsi_score'] = df['valley_rsi_diff'].abs()
    df['day_open_var'] = df['close']/df['close'].to_list()[0] - 1
    df['open_from_prev_close_var'] = df['close'].to_list()[0]/df['prev_close'] - 1
    df['ceil'] = df['close'].cummax()
    df['ceil_var'] = df['close']/df['ceil'] - 1
    df['floor'] = df['close'].cummin()
    df['floor_var'] = df['close']/df['floor'] - 1
    df['sym'] = sym
    #df['hour_of_day'] = (df['datetime'] - pd.Timedelta(minutes=29)).dt.hour
    #df['weekday'] = df['datetime'].dt.weekday.astype('category') #monday is 0
    return df

def add_is_profit(df, target_profit, target_loss):
    '''Returns Dataframe with additional columns, calculated based on input profit/loss parameters:
        actual_buy_price (float)
        profit (float)
        is_profit (bool)
    Args:
        df (pandas.DataFrame): Sorted Dataframe with at least these columns:
            close (float)
            divergence (str)
        target_profit (float): Target percentage profit e.g. 0.01 -> 1%
        target_loss (float): Target percentage loss e.g. 0.01 -> 1%
    Returns:
        df (pandas.DataFrame)
    '''
    buy_delay = 2 #only buy after n mins
    df['actual_buy_price'] = df['close'].shift(-buy_delay)
    df['profit'] = None
    for idx_div_row in df.index[df['divergence']!='']:
        actual_buy_price = df.iloc[idx_div_row, df.columns.get_loc('actual_buy_price')]
        profit = 0
        for selling_price in df.iloc[idx_div_row:-buy_delay, df.columns.get_loc('actual_buy_price')]:
            profit = (selling_price/actual_buy_price)-1
            if profit>target_profit or profit<target_loss:
                break
        df.at[idx_div_row, 'profit'] = profit
    df['is_profit'] = df['profit']>=target_profit
    df['profit'] = df['profit'].astype('float')
    return df

def get_dt_day_indicators(sym, close_latest, date_str_tdy, db):
    '''Returns dictionary with day level indicators for input symbol
    Args:
        sym (str)
        close_latest (float)
    Returns:
        dt_day_indicators (Dictionary)
    '''
    q='''
    with t as (
        select date, adj_close
          from prices_d
         where sym = '{}'
           and date(date) < '{}'
         order by date desc
         limit 185
    )
    select adj_close
      from t
     order by date
    '''.format(sym.upper(), date_str_tdy)
    df = pd.read_sql(q, db.conn)
    df = df.append(pd.DataFrame({'adj_close':[close_latest]}))
    df['sma9'] = df['adj_close'].rolling(9).mean()
    df['sma90'] = df['adj_close'].rolling(90).mean()
    df['sma180'] = df['adj_close'].rolling(180).mean()
    df['sma180'] = df['sma180'].fillna(df['sma90'])
    df['sma9_var'] = (df['adj_close']/df['sma9'])-1
    df['sma180_var'] = (df['adj_close']/df['sma180'])-1
    df = add_rsi(df, 14)
    ls_col = [
        'sma9_var',
        'sma180_var',
        'rsi14',
    ]
    if df[ls_col].iloc[-1].isnull().any():
        raise Exception(ERROR_NULL_DAY_LEVEL_IND)
    dt_day_indicators = dict(df[ls_col].iloc[-1])
    return dt_day_indicators

def add_day_level_indicators(df_i, sym, db):
    '''Returns df_interim with day-level indicators added
    Args:
        df_i (pandas.DataFrame)
        sym (str)
    Returns:
        df_i (pandas.DataFrame)
    '''
    close_latest = df_i['close'].to_list()[0]
    date_str_tdy = df_i['datetime'].to_list()[-1].strftime('%Y-%m-%d')
    dt_day_indicators = get_dt_day_indicators(sym, close_latest, date_str_tdy, db)
    for col, value in dt_day_indicators.items():
        df_i[f'day_{col}'] = value
    return df_i

def get_df_c(sym, date_str, live_data, db, target_profit, target_loss, index_limit=1000):
    '''Returns df_cooked
    Args:
        sym (str)
        date_str (str)
        live_data (int)
        db (DataBase object)
        target_profit (float): Target percentage profit e.g. 0.01 -> 1%
        target_loss (float): Target percentage loss e.g. -0.01 -> -1%
    Returns:
        df_c (pd.DataFrame)
    '''
    assert target_profit>0 and target_loss<0
    df_i = get_df_i(sym, date_str, live_data, db)
    df_i = df_i.iloc[:index_limit]
    df_i = add_peaks_valleys(df_i, order=5)
    df_i = add_valley_variances(df_i)
    df_i = add_divergences(df_i)
    df_i = add_day_level_indicators(df_i, sym, db)
    df_c = add_additional_measures(df_i, sym)
    df_c = add_is_profit(df_c, target_profit, target_loss)
    return df_c

def get_curr_price(sym):
    '''Returns current price for input symbol
    Args:
        sym (str)
    Returns:
        curr_price (float)
    '''
    df = yf.download(sym, period='1d', interval='1m', progress=0).reset_index()
    curr_price = df['Adj Close'].to_list()[-1]
    return curr_price

def get_df_info(sym):
    '''Returns dataframe containing general info about input symbol
    Args:
        sym (str): e.g.  BYND
    Returns:
        df_info (pandas.DataFrame)
            sym (str)
            long_name (str)
            sec (str)
            ind (str)
            quote_type (str)
            fund_family (str)
            summary (str)
            timestamp (datetime)
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

def check_prices_d_updated(date_str_tdy, db):
    '''Raises exception if table prices_d not updated
    Args:
        date_str_tdy (str)
        db (DataBase object)
    '''
    df = yf.download('IBM', period = '5d', interval='1d', progress=0).reset_index()
    date_str_last = df[df['Date']<date_str_tdy]['Date'].dt.date.astype('str').to_list()[-1]
    q = '''
        SELECT DATE(MAX(date)) AS date
          FROM prices_d
         WHERE DATE(date) < '{}'
    '''.format(date_str_tdy)
    df = pd.read_sql(q, db.conn)
    date_str_latest = df['date'].values[0]
    if date_str_latest != date_str_last:
        raise Exception(ERROR_PRICES_D_NOT_UPDATE.format(date_str_latest))