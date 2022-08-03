# Install packages
from azureml.core import Workspace, Dataset

# Define workspace
ws = Workspace.from_config()

# Retreive datasest from url and convert to dataframe
dataset = Dataset.Tabular.from_delimited_files(
    path='https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'
    , header=False
).to_pandas_dataframe()

# Change the column names
dataset.columns = [
    'WifesAge'
    , 'WifesEducation' # Low to High
    , 'HusbandsEducation' # Low to High
    , 'NumberOfChildren'
    , 'WifesReligion'  # Non-Islam/Islam
    , 'WifeWorking'  # Yes/No
    , 'HusbandOccupation'
    , 'StandardOfLivingIndex' # Low to High
    , 'MediaExposure' # Good/Not Good
    , 'ContraceptiveMethod'  # None/Long-Term/Short-Term
]

# Get datastore
datastore = ws.get_default_datastore()

# Register the dataframe
Dataset.Tabular.register_pandas_dataframe(
    dataframe=dataset
    , description='1987 National Indonesia Contraception Prevalence Survey'
    , name='1987_NICP_Survey'
    , show_progress=True
    , target=(datastore,'dataframe_data')
)