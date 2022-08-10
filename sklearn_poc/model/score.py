import joblib
import cloudpickle
from azureml.core import Model
import pandas as pd
import numpy as np

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
    
def init():    
    global preprocess, model
    # Load preprocessing
    preprocess_path = Model.get_model_path(model_name='sklearn_preprocessing')
    print('Preprocessing path is ', preprocess_path)
    with open(preprocess_path, mode='rb') as file:
        preprocess = cloudpickle.load(file)

    # Load model
    model_path = Model.get_model_path(model_name='sklearn_logistic')
    print('Model path is ', model_path)
    model = joblib.load(model_path)

data_sample = PandasParameterType(
    pd.DataFrame(
        {'WifesAge': pd.Series([0], dtype='int64')
        , 'WifesEducation': pd.Series(['example_value'], dtype='object')
        , 'HusbandsEducation': pd.Series(['example_value'], dtype='object')
        , 'NumberOfChildren': pd.Series([0], dtype='int64')
        , 'WifesReligion': pd.Series([0], dtype='int64')
        , 'WifeWorking': pd.Series(['example_value'], dtype='object')
        , 'HusbandOccupation': pd.Series(['example_value'], dtype='object')
        , 'StandardOfLivingIndex': pd.Series(['example_value'], dtype='object')
        , 'MediaExposure': pd.Series(['example_value'], dtype='object')
        }
    )
)

input_sample = StandardPythonParameterType({'data': data_sample})
result_sample = NumpyParameterType(np.array([0]))
output_sample = StandardPythonParameterType({'Results': result_sample})

@input_schema('Inputs', input_sample)
@output_schema(output_sample)

def run(Inputs):
    try:
        # Load from json
        data = Inputs['data']
        # Run through preprocessing pipeline
        processed_data = preprocess.transform(data)
        # Run processed_data through model
        result = model.predict_proba(processed_data)

        return {'data': result.to_list(), 'message': 'Successfully created predictions'}
    except Exception as err:
        error = str(err)
        return {'data': error, 'message': 'Failed to create predictions'}
