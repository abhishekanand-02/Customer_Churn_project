import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging 
from src.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

# Function to evaluate models based on R2 score
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        # Loop over models and evaluate each one
        for i in range(len(models)):
            model = list(models.values())[i]
            
            # Train the model
            model.fit(X_train, y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test)

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