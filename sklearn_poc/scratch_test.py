#%%
from azureml.core import (
    Workspace
    , Datastore
    , Dataset
)
import os
import numpy as np

# Define workspace from run.experiment
ws = Workspace.from_config()

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

# Get the input dataset by ID
dataset = Dataset.File.from_files(
    path=(default_datastore, 'Azure_ML_SDK1/sklearn_poc/data/train.txt')
)

# Create folder to store input pkl
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

# Download the input pkl into folder
dataset_dl = dataset.download(data_folder, overwrite=True)

train = np.loadtxt(
    fname=dataset_dl[0]
    , dtype=float
    )
# %%
np.shape(train)
# %%

