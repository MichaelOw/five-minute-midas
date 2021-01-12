import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
DIR_IMAGES = os.path.join(os.getcwd(), 'data', 'images')

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