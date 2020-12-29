import datetime
import pandas as pd

ls_d_str_holiday = [
    '2020-01-01',
    '2020-01-20',
    '2020-02-17',
    '2020-04-10',
    '2020-05-25',
    '2020-07-03',
    '2020-09-07',
    '2020-11-26',
    '2020-12-25',
    '2021-01-01',
    '2021-01-18',
    '2021-02-15',
    '2021-04-02',
    '2021-05-31',
    '2021-07-05',
    '2021-09-06',
    '2021-11-25',
    '2021-12-24',
    '2022-01-17',
    '2022-02-21',
    '2022-04-15',
    '2022-05-30',
    '2022-07-04',
    '2022-09-05',
    '2022-11-24',
    '2022-12-26',
    '2023-01-02',
    '2023-01-16',
    '2023-02-20',
    '2023-04-07',
    '2023-05-29',
    '2023-07-04',
    '2023-09-04',
    '2023-11-23',
    '2023-12-25',
]

def prev_weekday(date_str):
    '''Return previous weekday that is not a holiday
    Args:
        Date string yyyy-mm-dd
    Returns:
        Date string yyyy-mm-dd of previous weekday that is not a holiday
    '''
    d = (datetime.datetime.strptime(date_str, '%Y-%m-%d')
         - datetime.timedelta(days=1))
    d_str=d.strftime('%Y-%m-%d')
    while d.weekday() > 4 or d_str in ls_d_str_holiday: # Mon-Fri are 0-4
        d -= datetime.timedelta(days=1)
        d_str = d.strftime('%Y-%m-%d')
    return d_str
    
def add_days(date_str, n):
    '''
    Returns new date string with n days added
    Format: %Y-%m-%d
    '''
    return (datetime.datetime.strptime(date_str, '%Y-%m-%d')
            + datetime.timedelta(days=n)
            ).strftime('%Y-%m-%d')

def is_weekday(date_str):
    '''
    Args:
        date_str (str): e.g. '2020-06-22'
    Return:
        _ (int): 1 if date is weekday, else 0
    '''
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday() <= 4

def get_ls_date_str_from_db(start, end, db):
    '''Returns trading dates found in price_m, from start to end date inclusive
    Args:
        start (str): e.g. '2020-10-10'
        end (str): e.g. '2020-10-10'
    Returns:
        ls_date_str (list of str)
    '''
    assert end>=start
    q = '''
        SELECT DISTINCT DATE(datetime) AS date
          FROM prices_m
         WHERE sym='IBM'
         ORDER BY DATE(datetime)
    '''
    ls_date_str = pd.read_sql(q, db.conn)['date'].to_list()
    ls_date_str = [x for x in ls_date_str if start<=x<=end]
    return ls_date_str