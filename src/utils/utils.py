import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging 
from src.exception import customexception
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# Function to save model or object to a file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Save the object as a pickle file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving the object: {str(e)}")
        raise customexception(e, sys)

# Function to evaluate models based on R2 score (or other metrics if needed)
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Impute missing values using SimpleImputer
        imputer = SimpleImputer(strategy="mean")  # Using mean imputation for simplicity
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        # Loop over models and evaluate each one
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train the model
            model.fit(X_train_imputed, y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test_imputed)

            # Calculate R2 score for the model
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        logging.info("Model evaluation complete.")
        return report

    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {str(e)}")
        raise customexception(e, sys)

# Function to load a model or object from a file
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            logging.info(f"Object loaded successfully from {file_path}")
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading the object: {str(e)}")
        raise customexception(e, sys)
    
class HandleLastInteraction(BaseEstimator, TransformerMixin):
    """Custom transformer to handle 'Last Interaction' column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Replace 'null' and 'unknown' with NaN and convert to numeric
        X['Last Interaction'] = X['Last Interaction'].replace(['null', 'unknown'], np.nan)
        X['Last Interaction'] = pd.to_numeric(X['Last Interaction'], errors='coerce')  # Convert non-numeric to NaN
        X['Last Interaction'] = X['Last Interaction'].fillna(X['Last Interaction'].median())  # Fill NaN with median
        return X
