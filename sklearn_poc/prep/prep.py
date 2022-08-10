from azureml.core import Run, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

import argparse
import cloudpickle

from common import preprocessing_pipeline

if __name__ == "__main__":
    # Ensuring logging is possible
    run = Run.get_context()    

    # Instantiate ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--test_proportion", type=float)
    parser.add_argument("--target", type=str)

    args = parser.parse_args()

    # Define workspace from run.experiment
    ws = run.experiment.workspace

    # Get the input dataset by ID
    dataset = Dataset.get_by_id(
        workspace=ws
        , id=args.input_data
    )

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
            1:0 # No use of contraceptives
            ,2:1 # Short-Term
            ,3:1 # Long-Term
        }
        , inplace = True
    )

    Y_Test.replace(
        to_replace={
            1:0 
            ,2:1
            ,3:1
        }
        , inplace = True
    )

    # Instantiate the PreProcessing Pipeline
    PreProcessing_Pipeline = preprocessing_pipeline.CreateProcessingPipeline(verbose=False)

    # Create dictionary of how columns types shoudl be transformed
    To_Type_Dict = {
        'WifesAge': 'Int64'
        , 'WifesEducation': object
        , 'HusbandsEducation': object
        , 'NumberOfChildren': 'Int64'
        , 'WifesReligion': object
        , 'WifeWorking': object 
        , 'HusbandOccupation': object
        , 'StandardOfLivingIndex': object 
        , 'MediaExposure': object 
        # Comment out the Target
        # , 'ContraceptiveMethod': object
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
    