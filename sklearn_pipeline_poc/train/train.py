from azureml.core import (
    Run
    )

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import joblib
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Instantiate ArgumentParser and create arguments
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int)
# parser.add_argument("--train_data", type=str)
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.3)
parser.add_argument('--max_depth', type=int, dest='max_depth', default=6)
parser.add_argument('--colsample_bytree', type=float, dest='colsample_bytree', default=1.0)
parser.add_argument('--gamma', type=float, dest='gamma', default=0)
parser.add_argument('--reg_lambda', type=float, dest='reg_lambda', default=1.0)
parser.add_argument('--reg_alpha', type=float, dest='reg_alpha', default=0.0)
parser.add_argument('--subsample', type=float, dest='subsample', default=1.0)
parser.add_argument('--max_bin', type=int, dest='max_bin', default=256)

args = parser.parse_args()

# Define workspace from run.experiment
ws = run.experiment.workspace

# Get the input dataset by name of PipelineData
train_path = run.input_datasets['train_data']

# Load train_data
train = np.loadtxt(
    fname=os.path.join(train_path, 'train.txt')
    , dtype=float
)

# Separate into X_Train, Y_Train
x_train = train[:, 1:]
y_train = train[:, 0]

# Instantiate model with hyperparameters
model = xgb.XGBClassifier(
    random_state=args.random_seed
    , objective='binary:logistic' 
    , verbosity=0
    , n_jobs=1
    , tree_method='hist'
    , predictor='cpu_predictor'
    , use_label_encoder=False

    , n_estimators=args.n_estimators
    , learning_rate=args.learning_rate
    , max_depth=args.max_depth
    , colsample_bytree=args.colsample_bytree
    , gamma=args.gamma
    , reg_lambda=args.reg_lambda
    , reg_alpha=args.reg_alpha
    , subsample=args.subsample
    , max_bin=args.max_bin
)

# Store the model cross validation scores
model_scores = cross_val_score(
    estimator=model
    , X=x_train
    , y=y_train
    , cv=5
    , scoring='roc_auc'
    , fit_params={
        # can't use early_stopping_rounds without a specified validation set
        # 'early_stopping_rounds': 50 
        # , 
        'eval_metric': 'auc'
        , 'verbose': False
    }
)

# Store the mean cross-validated score
run.log('mean_cv_score', float(model_scores.mean()))

# Fit the model after cross_val_score
model.fit(
    X=x_train
    , y=y_train
)

# Store the model and complete the run
joblib.dump(
    value=model
    , filename="./outputs/model.pkl"
)