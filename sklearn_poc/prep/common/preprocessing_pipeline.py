import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler
    , OneHotEncoder
)
from sklearn.compose import (
    ColumnTransformer
    , make_column_selector
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from preprocessing_custom_transformers import (
    TypeConversion_Manual
    , FindCorrelation
    , CategoricalImputer
    , ModifiedLabelEncoder
    , FeatureMissingness
    , CategoryConverter
    , NewCategoryLevels
)

# This will allow a new Pipeline object to be instantiated without persisting connections to the underlying object
def CreateProcessingPipeline(verbose = False):
    # Define Numeric transformations
    Numeric_Transformer = Pipeline(
        steps=[
            ('SimpleImputer', SimpleImputer(
                strategy='median'
                , verbose=1))
            # No need for single numeric variable | commenting out FindCorrelation and VIFScreen
            # , ('CorrelationCheck', FindCorrelation(threshold=0.95))            
            # , ('Run VIF screen' ,VIFScreen(threshold = 10.0))
            , ('StandardScaler', StandardScaler())
        ]
        , verbose=verbose
    )

    # Define Numeric transformations
    Categorical_Transformer = Pipeline(
        steps=[
            ('CategoricalImputer', CategoricalImputer(
                strategy='most_frequent'))
            , ('LabelEncoder', ModifiedLabelEncoder())
            , ('OneHotEncoding', OneHotEncoder(
                handle_unknown='error'
                , drop='first'
                , sparse=False))
        ]
        , verbose=verbose
    )

    # Combine the Numeric and Categorical transformers
    Combined_Transformer = ColumnTransformer(
        transformers=[
            ('NumericTransforms'
             , Numeric_Transformer
             , make_column_selector(dtype_include=np.number))
            , ('CategoricalTransforms'
               , Categorical_Transformer
               , make_column_selector(dtype_include='category'))
        ]
        , remainder='passthrough'
    )

    # Inherit the Pipeline_PreProcessing object
    Processing_Pipeline = Pipeline(
        steps = [
            ('Convert_Column_Types', TypeConversion_Manual(column_type_dict={}))
            , ('Remove_High_Missing', FeatureMissingness(cutoff=0.90))
            , ('Convert_Object_To_Category', CategoryConverter())
            , ('Coerce_Novel_Levels', NewCategoryLevels())
            , ('Combined_Transforms', Combined_Transformer)
            , ('Variance_Screen', VarianceThreshold(threshold = 0))
        ]
        , verbose=verbose
    )

    # Return the final object
    return Processing_Pipeline