import os
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
DIR_IMAGES = os.path.join(os.getcwd(), 'data', 'images')

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
    if df.empty:
        ls_col = ['sym', 'datetime', 'my_index', 'proba', 'datetime_update']
        df_proba = pd.DataFrame(columns=ls_col)
    else:
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

def get_ls_col(tf, X):
    '''Get final list of columns from ColumnTransformer object
    Args:
        tf (ColumnTransformer)
        X (pandas.DataFrame)
    Returns:
        ls_col (List of str)
    '''
    ls_col = []
    for transformer in tf.transformers_:
        try:
            ls_col_append = list(transformer[1].get_feature_names())
        except:
            ls_col_append = list(transformer[2])
            if transformer[0] == 'remainder': ls_col_append = [list(X)[i] for i in ls_col_append]
        ls_col = ls_col + ls_col_append
    return ls_col

def plot_pr_curve(y, y_scores):
    '''Plot precision recall curve
    Args:
        y (pandas.Series): Series of booleans
        y_scores (numpy.ndarray): Array of scores
    '''
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center right')
    plt.ylim([0, 1])
    plt.show()

def save_pr_curve(y, y_scores, f='pr_curve.png'):
    '''Saves precision recall curve as png file
    Args:
        y (pandas.Series): Series of booleans
        y_scores (numpy.ndarray): Array of scores
    '''
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center right')
    plt.ylim([0, 1])
    plt.savefig(os.path.join(DIR_IMAGES, f), bbox_inches='tight')
    plt.clf()

def save_feature_importance(columns, feature_importances, f='feature_importance.png'):
    '''Saves feature importance chart as png file
    Args:
        columns (pandas.Series): Series of booleans
        feature_importances (numpy.ndarray): Array of importance scores
    '''
    df_fi = pd.DataFrame({
        'feature':columns,
        'importance':feature_importances
    }).sort_values('importance', ascending=0)
    ax = sns.barplot(x='importance', y='feature', data=df_fi)
    plt.savefig(os.path.join(DIR_IMAGES, f), bbox_inches='tight')
    plt.clf()