import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ML_Project.utils import save_oject

from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging

import os

@ dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this fuction is responsible for data transformation
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Loading train and test data file')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            
            # divide the train dataset to independent and dependent feature

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            tager_feature_train_df = train_df[target_column_name]

            # divide the test dataset to independent and dependent feature

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            tager_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing on trainnig and test dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Concatenation tain and test data

            train_arr = np.c_[
                input_feature_train_arr, np.array(tager_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(tager_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_oject(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)


