from azureml.core import (
    Workspace
    , ScriptRunConfig
    , Experiment
    , Environment
    , Dataset
    )
from azureml.core.compute import ComputeInstance
from azureml.core import Model

# from azureml.core.conda_dependencies import CondaDependencies

import os

# Construct workspace
ws = Workspace.from_config() 

# Define experiment
exp = Experiment(workspace=ws, name='my_sklearn_exp')

# Failed with image build and pip subprocess "could not find a version that satisfies the requirement azureml-samples==0+unknown (from -r /azureml-environment-setup/condaenv.um2xx4eg.requirements.txt"
# env = Environment.from_existing_conda_environment(    
#     name='test-env'
#     , conda_environment_name='azureml_py38'
# )

# # Bring in environment | doesn't work, won't import statsmodels... I can't figure out why
# env = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu")
# curated_clone = env.clone("customized_curated")
# conda_dep = CondaDependencies()
# conda_dep.add_conda_package("statsmodels")
# # conda_dep.add_pip_package("statsmodels")
# curated_clone.python.user_managed_dependencies=True
# curated_clone.python.conda_dependencies = conda_dep

# Build environment
env = Environment.from_conda_specification(name='test_env', file_path="conda_dependencies.yml")

# Define dataset
NICP_Survey = Dataset.get_by_name(
        workspace=ws
        , name="1987_NICP_Survey"
    )

# Define variables
random_seed = 123
test_proportion = 0.2
target = 'ContraceptiveMethod'

# Set the compute instance
instance = ComputeInstance(workspace=ws, name='crcastillo841')

# Structure the ScriptRunConfig
config = ScriptRunConfig(
    source_directory='./prep'
    , script='prep.py'
    # , environment=curated_clone
    , arguments=[
        '--input_data', NICP_Survey.as_named_input('NICP_Survey_Raw')
        , '--random_seed', random_seed
        , '--test_proportion', test_proportion
        , '--target', target
        ]
    , environment=env
    , compute_target=instance
    )

# Submit run and wait_for_completion
run = exp.submit(config)
run.wait_for_completion(show_output=True)

# Create a model folder in the current directory
os.makedirs('./prep/outputs', exist_ok=True)

# Download the PreProcessing model from run history
run.download_file(
    name='outputs/PreProcessing_Pipeline.cloudpkl'
    , output_file_path='./prep/outputs/PreProcessing_Pipeline.cloudpkl'
)

# Download train, test to ./data
for i in ['train', 'test']:
    run.download_file(
        name='outputs/' + str(i) + '.txt'
        , output_file_path='./data/' + str(i) + '.txt'
    )

# Register the PreProcessing Pipeline as model
run.register_model( 
    model_name='sklearn_preprocessing'
    , model_path='outputs/PreProcessing_Pipeline.cloudpkl' # run outputs path
    , description='A Pre-Processing Pipeline for sklearn models based on 1987 NICP Survey'
    , tags={
        'data-format': 'CSV'
        , 'dataset': '1987_NICP_Survey'
        }
    , model_framework=Model.Framework.SCIKITLEARN
    , model_framework_version='0.24.2'
    )