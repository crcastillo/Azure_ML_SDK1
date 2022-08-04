from azureml.core import (
    Experiment
    , Workspace
    , Environment
    , Dataset
    , Datastore
    , ScriptRunConfig
    )
from azureml.core.compute import ComputeInstance
from azureml.train.hyperdrive import (
    HyperDriveConfig
    , PrimaryMetricGoal
    , BayesianParameterSampling
    , uniform
    , loguniform
    )

import numpy as np

# Construct workspace
ws = Workspace.from_config() 

# Define experiment
exp = Experiment(workspace = ws, name = 'sklearn_hyperdrive_training')

# Build environment
env = Environment.from_conda_specification(name='test_env', file_path="conda_dependencies.yml")

# Set the compute instance
instance = ComputeInstance(workspace=ws, name='crcastillo841')

# Define search space and parameter search
param_sampling = BayesianParameterSampling({
    "l1_ratio": uniform(
        min_value=0
        , max_value=1
        ),
    "C": loguniform(
        min_value=1e-4
        , max_value=1e2
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
    , pattern='*train.pkl'
)

'''
LEFT OFF HERE
'''

# Define dataset
train = Dataset.from_binary_files(default_datastore.path('./data/train.pkl'))

# Define variables
random_seed = 123

# Structure the ScriptRunConfig
run_config = ScriptRunConfig(
    source_directory='./train'
    , script='train.py'
    , arguments=[
        # '--input_data', train.as_named_input('train_data')
        # , 
        '--random_seed', random_seed
        ]
    , environment=env
    , compute_target=instance
    , 
)

# Assumes ws, script_config and param_sampling are already defined
hyperdrive = HyperDriveConfig(
    run_config=run_config
    , hyperparameter_sampling=param_sampling
    , policy=None
    , primary_metric_name='mean_cv_score'
    , primary_metric_goal=PrimaryMetricGoal.MAXIMIZE
    , max_total_runs=6
    , max_concurrent_runs=4
    )

# Submit run and wait_for_completion
hyperdrive_run = exp.submit(config=hyperdrive)
hyperdrive_run.wait_for_completion(show_output=True)