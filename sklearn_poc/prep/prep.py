from azureml.core import Run, Dataset, Workspace
from sklearn.model_selection import train_test_split

import argparse
import joblib

from common import preprocessing_pipeline

if __name__ == "__main__":
    # Ensuring logging is possible
    run = Run.get_context()

    # Set variables
    random_seed = 123
    test_proportion = 0.2
    target = 'ContraceptiveMethod'

    # Instantiate ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
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
        raw_df.loc[:, ~raw_df.columns.isin([target])]
        , raw_df.loc[: , raw_df.columns.isin([target])]
        , random_state=random_seed
        , test_size=test_proportion
    )

    # Fix Y_Train and Y_Test
    Y_Train.replace(
        to_replace={3:1}
        , inplace = True
    )

    Y_Test.replace(
        to_replace={3:1}
        , inplace = True
    )

    # Instantiate the PreProcessing Pipeline
    PreProcessing_Pipeline = preprocessing_pipeline.CreateProcessingPipeline(verbose=False)

    # Create dictionary of how columns types shoudl be transformed
    To_Type_Dict = {
        'WifesAge': 'Int64'
        , 'WifesEducation': object
        , 'HusbandsEducation': object
        , 'NumberOfChildren': object
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

    # Create dict of objects for pkl
    pkl_dict = {
        'X_Train_trans': X_Train_trans
        , 'Y_Train': Y_Train
        , 'X_Test_trans': X_Test_trans
        , 'Y_Test': Y_Test
        , 'PreProcessing_Pipeline': PreProcessing_Pipeline
    }

    # Iteratively dump objects into ./outputs
    for i in pkl_dict:
        joblib.dump(
            value=pkl_dict[i]
            , filename='./outputs/' + str(i) + '.pkl'
        )