from azureml.core import (
    Experiment
    , Workspace
    , Environment
    , Dataset
    , Datastore
    , ScriptRunConfig
    , Model
    )
from azureml.core.compute import ComputeTarget
from azureml.train.hyperdrive import (
    HyperDriveConfig
    , PrimaryMetricGoal
    , BayesianParameterSampling # Bayesian doesn't support loguniform
    , uniform
    , choice
    )

import numpy as np
import os

# Define variables
random_seed = 123

# Construct workspace
ws = Workspace.from_config() 

# Define experiment
exp = Experiment(workspace = ws, name = 'sklearn_hyperdrive_training')

# Build environment
env = Environment.from_conda_specification(name='test_env', file_path="conda_dependencies.yml")

# Set the compute target (cluster)
cluster = ComputeTarget(workspace=ws, name='cpu-cluster')

# Define search space and parameter search
param_sampling = BayesianParameterSampling({
    "l1_ratio": uniform(
        min_value=0
        , max_value=1
        ),
    "C": choice(
        [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
        )
})

# Configure the default storage
default_datastore = Datastore.get(
    workspace=ws
    , datastore_name='workspaceblobstore'
)

# Upload train data
Dataset.File.upload_directory(
    src_dir='./data/'
    , target=(default_datastore, 'Azure_ML_SDK1/sklearn_poc/data/')
    , overwrite=True
    , pattern='*train.txt'
)

# Structure the ScriptRunConfig
run_config = ScriptRunConfig(
    source_directory='./train'
    , script='train.py'
    , arguments=[
        '--random_seed', random_seed
        ]
    , environment=env
    , compute_target=cluster
)

# Assumes ws, script_config and param_sampling are already defined
hyperdrive = HyperDriveConfig(
    run_config=run_config
    , hyperparameter_sampling=param_sampling
    , policy=None
    , primary_metric_name='mean_cv_score'
    , primary_metric_goal=PrimaryMetricGoal.MAXIMIZE
    , max_total_runs=40
    , max_concurrent_runs=4
    )

# Submit run and wait_for_completion
hyperdrive_run = exp.submit(config=hyperdrive)
hyperdrive_run.wait_for_completion(show_output=True)

# Print best_run
best_run = hyperdrive_run.get_best_run_by_primary_metric()
print(best_run)

# Create a model folder in the current directory
os.makedirs('./model/outputs', exist_ok=True)

# Download the PreProcessing model from run history
best_run.download_file(
    name='outputs/model.pkl'
    , output_file_path='./model/outputs/model.pkl'
)

# Register best model
best_run.register_model(
    model_name='sklearn_logistic'
    , model_path='outputs/model.pkl' # run outputs path
    , description='A LogisticRegression for 1987 NICP Survey'
    , tags={
        'data-format': 'CSV'
        , 'dataset': '1987_NICP_Survey'
        , 'type': 'classification'
        , 'model_form': 'logistic regression'
        }
    , model_framework=Model.Framework.SCIKITLEARN
    , model_framework_version='0.24.2'
)