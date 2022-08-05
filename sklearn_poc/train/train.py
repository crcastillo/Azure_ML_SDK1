from azureml.core import (
    Run
    , Dataset
    , Datastore
    )

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

import numpy as np
import joblib
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Instantiate ArgumentParser and create arguments
parser = argparse.ArgumentParser()
# parser.add_argument("--input_data", type=str)
parser.add_argument('--C', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--l1_ratio', type=float, dest='l1_ratio', default=0)
parser.add_argument("--random_seed", type=int)

args = parser.parse_args()

# Define workspace from run.experiment
ws = run.experiment.workspace

# Configure the default storage
default_datastore = Datastore.get(
    workspace=ws
    , datastore_name='workspaceblobstore'
)

# Get the input dataset by ID
dataset = Dataset.File.from_files(
    path=(default_datastore, 'Azure_ML_SDK1/sklearn_poc/data/train.txt')
)

# Create folder to store input pkl
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

# Download the input pkl into folder
dataset_dl = dataset.download(data_folder, overwrite=True)

# Load to numpy array
train = np.loadtxt(
    fname=dataset_dl[0]
    , dtype=float
)

# Separate into X_Train, Y_Train
x_train = train[:, 1:]
y_train = train[:, 0]


# Instantiate model with hyperparameters
model = LogisticRegression(
    penalty='elasticnet' # Allows full range of penalization
    , solver='saga'  # Only solver to support range of l1, l2 penalties
    , random_state=args.random_seed

    , C=args.reg_rate
    , l1_ratio=args.l1_ratio
)

# Store the model cross validation scores
model_scores = cross_val_score(
    estimator=model
    , X=x_train
    , y=y_train
    , cv=5
    , scoring='roc_auc'
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

# Finish the run        
run.complete()