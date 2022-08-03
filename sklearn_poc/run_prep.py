from azureml.core import (
    Workspace
    , ScriptRunConfig
    , Experiment
    , Environment
    , Dataset
    )
from azureml.core.compute import ComputeInstance
# from azureml.core.conda_dependencies import CondaDependencies


ws = Workspace.from_config(path='./config.json') 
exp = Experiment(workspace=ws, name='my_exp')

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

instance = ComputeInstance(workspace=ws, name='crcastillo841')
config = ScriptRunConfig(
    source_directory='.'
    , script='./prep/prep.py'
    # , environment=curated_clone
    , arguments=['--input-data', NICP_Survey.as_named_input('NICP_Survey_Raw')]
    , environment=env
    , compute_target=instance
    )
run = exp.submit(config)
run.wait_for_completion(show_output=True)