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
parser.add_argument('--preprocess_data', type=str, dest='preprocess_data', help='path to preprocess pipeline file')
args = parser.parse_args()

# Define workspace from run.experiment
ws = run.experiment.workspace

# Location of model for registration
model_output_dir = './register_preprocess/'

# Create the directory and copy 
os.makedirs(model_output_dir, exist_ok=True)
copy2(
    src=os.path.join(args.preprocess_data, "PreProcessing_Pipeline.cloudpkl")
    , dst=model_output_dir
)

# Register the model
model = Model.register(
    workspace=ws
    , model_name='adult_pipeline_preprocessing'
    , model_path=model_output_dir
    , description='A Pre-Processing Pipeline for xgboost model based on 1994 Adult Data'
    , tags={
        'data-format': 'CSV'
        , 'dataset': '1994_Adult_Data'
        }
    , model_framework=Model.Framework.SCIKITLEARN
    , model_framework_version='0.24.2'
    )