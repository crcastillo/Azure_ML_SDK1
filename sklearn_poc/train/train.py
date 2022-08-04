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

'''
LEFT OFF HERE
'''

# Separate into X_Train, Y_Train
# X_Train = train[:, 0]
# Y_train = [:, 1:]

# Instantiate a record of the cv scores
best_score = None

# Iterate through search_grid rows
for i in search_grid.index:
    # Start logging
    run = experiment.start_logging()

    # Capture hyperparamters
    # run.log('C', search_grid['C'].loc[i])
    run.log('l1_ratio', search_grid['l1_ratio'].loc[i])

    # Instantiate model with hyperparameters
    model = LogisticRegression(
        penalty='elasticnet' # Allows full range of penalization
        , solver='saga'  # Only solver to support range of l1, l2 penalties
        , random_state=random_seed

        , C=args.reg_rate
        , l1_ratio=args.l1_ratio
    )

    # Store the model cross validation scores
    model_scores = cross_val_score(
        estimator=model
        , X=X_Train_trans
        , y=Y_Train.values.ravel()
        , cv=5
        , scoring='roc_auc'
    )

    # Store the mean cross-validated score
    run.log('mean_cv_score', model_scores.mean())

    # Boolean check as whether to save new best model
    if best_score == None or model_scores.mean() > best_score:
        # Take the new best_score
        best_score = model_scores.mean()

        # Fit the model
        model.fit(
            X=X_Train_trans
            , y=Y_Train.values.ravel()
        )

        # Store the model name for export
        model_name = "model_version_" + str(i) + ".pkl"

        # Store the model and complete the run
        filename = "outputs/" + model_name
        joblib.dump(
            value=model
            , filename=filename
            )
        run.upload_file(
            name=model_name
            , path_or_stream=filename
            )

    # Finish the run        
    run.complete()



# load the training dataset
data = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()