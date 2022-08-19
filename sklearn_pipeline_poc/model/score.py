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
    preprocess_path = Model.get_model_path(model_name='adult_pipeline_preprocessing')
    print('Preprocessing path is ', preprocess_path)
    with open(preprocess_path, mode='rb') as file:
        preprocess = cloudpickle.load(file)

    # Load model
    model_path = Model.get_model_path(model_name='adult_pipeline_xgboost')
    print('Model path is ', model_path)
    model = joblib.load(model_path)

data_sample = PandasParameterType(
    pd.DataFrame(
        {
        'age': pd.Series([39], dtype='int64')
        , 'workclass': pd.Series(['State-gov'], dtype='object')
        , 'fnlwgt': pd.Series([77516], dtype='int64')
        , 'education': pd.Series(['Bachelors'], dtype='object')
        , 'education_num': pd.Series([13], dtype='int64')
        , 'marital_status': pd.Series(['Never-married'], dtype='object')
        , 'occupation': pd.Series(['Adm-clerical'], dtype='object')
        , 'relationship': pd.Series(['Not-in-family'], dtype='object')
        , 'race': pd.Series(['White'], dtype='object')
        , 'sex': pd.Series(['Male'], dtype='object')
        , 'capital_gain': pd.Series([2714], dtype='int64')
        , 'capital_loss': pd.Series([0], dtype='int64')
        , 'hours_per_week': pd.Series([40], dtype='int64')
        , 'native_country': pd.Series(['United States'], dtype='object')
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

        return {'data': result.tolist(), 'message': 'Successfully created predictions'}
    except Exception as err:
        error = str(err)
        return {'data': error, 'message': 'Failed to create predictions'}
