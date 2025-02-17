import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
import os
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.utils import save_object
from src.utils.utils import HandleLastInteraction


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    processed_train_file_path = os.path.join('artifacts', 'processed_train.csv')
    processed_test_file_path = os.path.join('artifacts', 'processed_test.csv')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            numerical_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
                                 'Total Spend', 'Last Interaction']  # Exclude 'Subscription Type' and 'Contract Length'

            categorical_columns = ['Gender', 'Contract Length', 'Subscription Type']

            logging.info('Pipeline Initiated')

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values with the median
                    ('scaler', StandardScaler())  # Standardize numerical features
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
                ]
            )

            # Apply the numerical and categorical pipelines to the respective columns
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_columns),
                ('categorical_pipeline', categorical_pipeline, categorical_columns),
                ('last_interaction_handler', HandleLastInteraction(), ['Last Interaction'])  # Add custom transformer here
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Exception occurred in get_data_transformation: {str(e)}")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test data.")
            
            preprocessing_pipeline = self.get_data_transformation()

            target_column = 'Churn'  
            columns_to_drop = ['CustomerID', target_column]  

            # Separate features (input) and target for both train and test data
            X_train = train_df.drop(columns=columns_to_drop, axis=1)  
            y_train = train_df[target_column]  

            X_test = test_df.drop(columns=columns_to_drop, axis=1)  
            y_test = test_df[target_column] 

            # Handle missing values for target variable (y_train, y_test)
            y_train = y_train.replace(["null", "unknown"], np.nan)  # Replace "null" and "unknown" with NaN
            y_test = y_test.replace(["null", "unknown"], np.nan)

            y_train = y_train.fillna(y_train.mode()[0])
            y_test = y_test.fillna(y_test.mode()[0])  

            # Apply the preprocessing pipeline to the input features
            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)

            logging.info("Preprocessing completed on training and testing datasets.")

            # Combine transformed features with target variable for both training and testing
            training_data = np.c_[X_train_transformed, np.array(y_train)]
            testing_data = np.c_[X_test_transformed, np.array(y_test)]

            # Save the processed data to CSV files
            processed_train_df = pd.DataFrame(training_data)
            processed_test_df = pd.DataFrame(testing_data)

            processed_train_df.to_csv(self.data_transformation_config.processed_train_file_path, index=False, header=False)
            processed_test_df.to_csv(self.data_transformation_config.processed_test_file_path, index=False, header=False)

            logging.info("Processed train and test data saved as CSV files.")

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
