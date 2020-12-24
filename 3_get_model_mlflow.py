import os
import time
import json
import mlflow
import winsound
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils_general import get_df_parquet
from src.utils_model import get_ls_col
from src.utils_model import plot_pr_curve
dir_train = os.path.join(os.getcwd(), 'data', 'train')
dir_models = os.path.join(os.getcwd(), 'data', 'models')
dir_mlflow = 'file:' + os.sep + os.path.join(os.getcwd(), 'mlflow')
mlflow.set_tracking_uri(dir_mlflow)

def beeps(n=3):
    '''Produce n beeps. Default 3'''
    for _ in range(n):
        winsound.Beep(200,500)

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]

class RandomForestClassifierFlow():
    def __init__(self, params={}, tags={}):
        self.model = RandomForestClassifier(**params)
        self.params = params
        tags['model'] = 'RandomForestClassifier'
        self.tags = tags

    def mlflow_run(self, df):
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id
            # train test split
            train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['is_profit']])
            y = train['is_profit'].copy()
            X = train.drop(columns=['is_profit']).copy()
            y_test = test['is_profit'].copy()
            X_test = test.drop(columns=['is_profit']).copy()
            # pipeline
            float_cols = df.select_dtypes(include='float64').columns
            preprocessor = ColumnTransformer([
                ('StandardScaler', StandardScaler(), float_cols),
                #('OneHotEncoder', OneHotEncoder(), cat_cols),
                ]
                ,remainder='passthrough')
            full_pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', self.model),])
            # fit
            t_start = time.time()
            full_pipe.fit(X, y)
            t_training = time.time() - t_start
            # predict
            t_start = time.time()
            y_test_pred_proba = full_pipe.predict_proba(X_test)
            t_prediction = time.time() - t_start
            # score
            proba_threshold = 0.75
            metrics = {
                'auroc':roc_auc_score(y_test, y_test_pred_proba[:,1]),
                'precision':precision_score(y_test, (y_test_pred_proba[:,1]>proba_threshold)),
                't_training':t_training,
                't_prediction':t_prediction,
            }
            # log params, metrics, tags
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(self.tags)
            # log Model
            #mlflow.sklearn.log_model(full_pipe, artifact_path='model')
            #wrapped_model = SklearnModelWrapper(full_pipe)
            #mlflow.pyfunc.log_model('model', python_model=wrapped_model)
            return full_pipe

#############
# Prep data #
#############
print('Available df_train files:')
[print(f"'{x}',") for x in os.listdir(dir_train) if x[-8:]=='.parquet'];
# df_train - Import
ls_f = [   
    'df_train_20201204_1216.parquet',
    'df_train_20201204_1219.parquet',
    'df_train_20201212_1545.parquet',
]
df = get_df_parquet(ls_f, dir_train)

# df_train - Remove outliers and non-relevant data 
q = '''
    divergence=='bull_reg'\
    and prev_close>5\
    and abs(sma9_var)<0.02\
    and abs(sma180_var)<0.2\
    and abs(vwap_var)<0.2\
    and abs(spread14_e)<0.02\
    and abs(prev_close_var)<0.5\
    and abs(prev_floor_var)<0.5\
    and abs(prev_ceil_var)<0.5\
    and abs(prev1_candle_score)<0.02\
    and abs(prev2_candle_score)<0.02\
    and abs(prev3_candle_score)<0.02\
    and mins_from_start<300\
    and valley_interval_mins<200\
    and valley_close_score<10\
    and abs(day_open_var)<1.5\
    and abs(open_from_prev_close_var)<0.4\
    and abs(ceil_var)<0.2\
    and abs(floor_var)<0.2\
'''
df = df.query(q)

# df_train - get dates
df = df[df['datetime'].dt.date.astype('str')>'2020-06-28']
inputs_date_start = df['datetime'].dt.date.astype('str').unique().min()
inputs_date_end = df['datetime'].dt.date.astype('str').unique().max()
print(inputs_date_start, inputs_date_end)

# df_train - Remove unwanted columns
ls_col_remove = [
    'sym',
    'datetime',
    'prev_close',
    'divergence',
    'profit',
    ###
    #'valley_interval_mins',
    #'floor_var',
    #'sma9_var',
    #'prev_close_var',
    #'ceil_var',
    #'prev_ceil_var',
]
df = df.drop(columns=ls_col_remove)
ls_col = list(df.drop(columns='is_profit'))

# df-train - Preview
df.info()

##################################
# Run test and track with MLflow #
##################################
params = {
    'criterion': 'entropy',
    'max_depth': 1000,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 5,
    'n_estimators': 600,
    ###
    'n_jobs': -1,
    'random_state': 42,
}
tags = {
    'inputs_date_start':inputs_date_start,
    'inputs_date_end':inputs_date_end,
    'df_train files':str(ls_f),
    'comments':''
}
rfcf = RandomForestClassifierFlow(params, tags)
full_pipe = rfcf.mlflow_run(df)

##############
# Save model #
##############
import pickle
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
tup_model = (q, ls_col, full_pipe)
f = os.path.join(dir_models, f'tup_model_{timestamp}.p')
pickle.dump(tup_model, open(f, 'wb'))
print(f, 'saved')