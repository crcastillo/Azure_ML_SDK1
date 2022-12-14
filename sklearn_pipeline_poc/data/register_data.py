# Install packages
from azureml.core import Workspace, Dataset

# Define workspace
ws = Workspace.from_config()

# Retreive datasest from url and convert to dataframe
dataset = Dataset.Tabular.from_delimited_files(
    path='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    , header=False
).to_pandas_dataframe()

# Change the column names
dataset.columns = [
    'age' # continuous
    , 'workclass' # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
    , 'fnlwgt' # continuous
    , 'education' # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
    , 'education_num' # continuous
    , 'marital_status' # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
    , 'occupation' # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
    , 'relationship' # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
    , 'race' # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
    , 'sex' # Female, Male
    , 'capital_gain' # continuous
    , 'capital_loss' # continuous
    , 'hours_per_week' # continuous
    , 'native_country' # United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
    , 'income' # >50K, <=50K
]

# Get datastore
datastore = ws.get_default_datastore()

# Register the dataframe
Dataset.Tabular.register_pandas_dataframe(
    dataframe=dataset
    , description='1994 Adult Data Set'
    , name='1994_Adult_Data'
    , show_progress=True
    , target=(datastore,'dataframe_data')
)