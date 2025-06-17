import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation process.
        '''
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns =[
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), # Handling missing values
                    ('scaler', StandardScaler()) # Feature scaling
                ]
            )
            logging.info("Numerical and categorical columns defined for preprocessing.")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), # Handling missing values
                    ('onehot', OneHotEncoder(handle_unknown='ignore')), # One-hot encoding
                    ('scaler', StandardScaler(with_mean=False)) # Scaling categorical features
                ]
            )
            logging.info("Numerical and categorical pipelines created for preprocessing.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error("Error occurred while creating data transformer object")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation...")
            preprocessor = self.get_data_transformer_object()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")

            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Input and target features separated for train and test datasets.")

            # Applying the preprocessor to the training and testing data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Data transformation applied to train and test datasets.")

            # Saving the preprocessor object
            train_arr = np.c_[ # Combine input features and target variable for training
                input_feature_train_arr, np.array(target_feature_train_df) # Combine input features and target variable for training
            ]
            test_arr = np.c_[ # Combine input features and target variable for testing
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Train and test arrays created successfully.")

            # Saving the preprocessor object to a file
            save_object( # fuction defined in src/utils.py
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys) from e