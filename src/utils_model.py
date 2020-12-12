import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

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