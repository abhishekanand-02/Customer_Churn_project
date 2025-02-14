import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.utils import save_object
from src.logger import logging
from src.exception import customexception
import sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        # Fit a separate encoder for each column
        for i in range(X.shape[1]):
            self.encoders[i] = LabelEncoder().fit(X[:, i])
        return self

    def transform(self, X):
        # Apply each encoder to its respective column
        X_encoded = X.copy()
        for i in range(X.shape[1]):
            # Handle unseen labels by using the same encoding as the most frequent label in the training data
            X_encoded[:, i] = self.encoders[i].transform(X[:, i])
        return X_encoded


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    transformed_train_file_path = os.path.join('artifacts', 'transformed_train.csv')
    transformed_test_file_path = os.path.join('artifacts', 'transformed_test.csv')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']
            categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']

            logging.info('Pipeline Initiated')

            # Pipeline for numerical data
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values with median
                    ('scaler', StandardScaler())  # Standardize numerical features
                ]
            )

            # Pipeline for categorical data
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical values
                    ('label_encoder', LabelEncoderTransformer())  # Apply label encoding to categorical columns
                ]
            )

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Exception occurred in get_data_transformation: {str(e)}")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            # Load training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test data.")

            # logging.info(f"Missing values in training data before removal:\n{train_df.isnull().sum()}")
            # logging.info(f"Missing values in testing data before removal:\n{test_df.isnull().sum()}")

            train_df = train_df.dropna()  
            test_df = test_df.dropna()  

            # logging.info(f"Missing values in training data after removal:\n{train_df.isnull().sum()}")
            # logging.info(f"Missing values in testing data after removal:\n{test_df.isnull().sum()}")

            preprocessing_pipeline = self.get_data_transformation()

            target_column = 'Churn'  
            columns_to_drop = ['CustomerID', target_column]

            X_train = train_df.drop(columns=columns_to_drop, axis=1)  
            y_train = train_df[target_column]  

            X_test = test_df.drop(columns=columns_to_drop, axis=1) 
            y_test = test_df[target_column]  

            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)
            
            logging.info("Preprocessing completed on training and testing datasets.")

            # Combine features and target into a single dataset
            training_data = np.c_[X_train_transformed, np.array(y_train)]
            testing_data = np.c_[X_test_transformed, np.array(y_test)]

            # Save transformed training and test data as CSV
            pd.DataFrame(training_data).to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            pd.DataFrame(testing_data).to_csv(self.data_transformation_config.transformed_test_file_path, index=False)

            logging.info("Transformed data saved as 'transformed_train.csv' and 'transformed_test.csv'.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )

            logging.info("Preprocessing object saved successfully.")

            return training_data, testing_data

        except Exception as e:
            logging.error(f"Exception occurred in initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)


if __name__ == "__main__":
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    data_transformation_instance = DataTransformation()
    train_data, test_data = data_transformation_instance.initialize_data_transformation(train_data_path, test_data_path)
