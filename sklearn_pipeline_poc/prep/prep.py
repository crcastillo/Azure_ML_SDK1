from azureml.core import Run, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

import os
import argparse
import cloudpickle

from common import preprocessing_pipeline

if __name__ == "__main__":
    # Ensuring logging is possible
    run = Run.get_context()    

    # Instantiate ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--test_proportion", type=float)
    parser.add_argument("--target", type=str)
    parser.add_argument("--train_data", dest="train_data", required=True)
    parser.add_argument("--test_data", dest="test_data", required=True)

    args = parser.parse_args()

    # Define workspace from run.experiment
    ws = run.experiment.workspace

    # Get the input dataset by ID
    dataset = run.input_datasets['ds_input']

    # load the TabularDataset to pandas DataFrame
    raw_df = dataset.to_pandas_dataframe()

    # Split Train_Data into Train/Valid
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        raw_df.loc[:, ~raw_df.columns.isin([args.target])]
        , raw_df.loc[: , raw_df.columns.isin([args.target])]
        , random_state=args.random_seed
        , test_size=args.test_proportion
    )

    # Fix Y_Train and Y_Test
    Y_Train.replace(
        to_replace={
            ' <=50K':0 # No use of contraceptives
            ,' >50K':1 # Short-Term
        }
        , inplace = True
    )

    Y_Test.replace(
        to_replace={
            ' <=50K':0 
            , ' >50K':1
        }
        , inplace = True
    )

    # Instantiate the PreProcessing Pipeline
    PreProcessing_Pipeline = preprocessing_pipeline.CreateProcessingPipeline(verbose=False)

    # Create dictionary of how columns types shoudl be transformed
    To_Type_Dict = {
        'age': 'Int64'
        , 'workclass': object
        , 'fnlwgt': 'Int64'
        , 'education': object
        , 'education_num': 'Int64'
        , 'marital_status': object
        , 'occupation': object
        , 'relationship': object
        , 'race': object
        , 'sex': object
        , 'capital_gain': 'Int64'
        , 'capital_loss': 'Int64'
        , 'hours_per_week': 'Int64'
        , 'native_country': object
        # Comment out the Target
        # , 'income' object
    }

    # Alter the parameters of pipeline
    PreProcessing_Pipeline.set_params(**{
        'Convert_Column_Types__column_type_dict': To_Type_Dict
    })

    # Transform X_Train for model fitting
    X_Train_trans = PreProcessing_Pipeline.fit_transform(X=X_Train)

    # Transform X_Test
    X_Test_trans = PreProcessing_Pipeline.transform(X=X_Test)

    # Column stack target and features
    train = np.column_stack((Y_Train, X_Train_trans))
    test = np.column_stack((Y_Test, X_Test_trans))

    # Save train/test as numpy arrays
    np.savetxt(
        fname='./outputs/train.txt'
        , X=train
        , fmt="%f"
    )
    np.savetxt(
        fname='./outputs/test.txt'
        , X=test
        , fmt="%f"
    )

    # Dump PreProcessing_Pipeline into ./outputs
    cloudpickle.register_pickle_by_value(preprocessing_pipeline)
    with open('./outputs/PreProcessing_Pipeline.cloudpkl', mode='wb') as file:
        cloudpickle.dump(
            obj=PreProcessing_Pipeline
            , file=file
        )
    
    # Export train to Pipeline data
    if not (args.train_data is None):
        os.makedirs(args.train_data, exist_ok=True)
        print("%s created" % args.train_data)
        path = args.train_data + "/train.txt"
        np.savetxt(
            fname=path
            , X=train
            , fmt="%f"
        )

    # Export test to Pipeline data
    if not (args.test_data is None):
        os.makedirs(args.test_data, exist_ok=True)
        print("%s created" % args.test_data)
        path = args.test_data + "/test.txt"
        np.savetxt(
            fname=path
            , X=test
            , fmt="%f"
        )