import argparse
import os
from azureml.core import (
    Model
    , Run
)
from shutil import copy2

# Ensure logging is possible
run = Run.get_context()

# Instantiate ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--model_data', type=str, dest='model_data', help='path to model file')
args = parser.parse_args()

# Define workspace from run.experiment
ws = run.experiment.workspace

# Location of model for registration
model_output_dir = './register_model/'

# Create the directory and copy 
os.makedirs(model_output_dir, exist_ok=True)
copy2(args.model_data, model_output_dir)

# Register the model
model = Model.register(
    workspace=ws
    , model_name='adult_pipeline_xgboost'
    , model_path=model_output_dir
    , description='A XGBoost model for 1994 Adult Income Data'
    , tags={
        'data-format': 'CSV'
        , 'dataset': '1994_Adult_Data'
        , 'type': 'classification'
        , 'model_form': 'xgboost'
        , 'labels': '0, 1'
        }
    , model_framework=Model.Framework.SCIKITLEARN
    , model_framework_version='0.24.2'
    )