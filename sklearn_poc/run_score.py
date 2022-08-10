from azureml.core import (
    Workspace
    , Environment
    )
from azureml.core.model import (
    InferenceConfig
    , Model
    )
from azureml.core.webservice import AciWebservice

# Define workspace
ws = Workspace.from_config()

# Build environment
env = Environment.from_conda_specification(name='test_env', file_path="conda_dependencies.yml")

# Register the environment
env.register(workspace=ws)

# Define the ACI configuration
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1
    , memory_gb=2
    , tags={
        'data':'1987 NICP Survey contraception classifier'
    }
    , description='Classification of contraception use'
)

# Create the inference configuration
inference_config = InferenceConfig(
    entry_script='score.py'
    , source_directory='./model'
    , environment=env
)

# Model deployment config
service = Model.deploy(
    workspace=ws
    , name='nicp-model-log'
    , models=[
        Model._get(
            workspace=ws
            , name='sklearn_preprocessing'
        )
        , Model._get(
            workspace=ws
            , name='sklearn_logistic'
        ) 
        ]
    , inference_config=inference_config
    , deployment_config=aci_config
    , overwrite = True
    )

# Deploy
service.wait_for_deployment(show_output=True)
url = service.scoring_uri
print(url)