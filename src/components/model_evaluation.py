import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from src.utils.utils import load_object
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.logger import logging
from src.exception import customexception
from urllib.parse import urlparse
import numpy as np

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, actual, pred):
        """Evaluate the model performance with RMSE, MAE, and R2"""
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))  # Root Mean Squared Error
            mae = mean_absolute_error(actual, pred)  # Mean Absolute Error
            r2 = r2_score(actual, pred)  # R2 Score
            logging.info(f"Evaluation metrics captured: RMSE={rmse}, MAE={mae}, R2={r2}")
            return rmse, mae, r2
        except Exception as e:
            logging.error("Error in calculating evaluation metrics")
            raise customexception(e, sys)

    def initiate_model_evaluation(self, train_array, test_array):
        """Initiate model evaluation and log metrics using MLflow"""
        try:
            # Splitting features and target from the test array
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "best_model.pkl")
            model = load_object(model_path)

            # Log the model evaluation process
            logging.info("Model loaded and evaluation initiated")

            # Get the tracking URL type for MLflow
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"Tracking URL type: {tracking_url_type_store}")

            with mlflow.start_run():
                # Predict using the model
                prediction = model.predict(X_test)

                # Evaluate model metrics
                rmse, mae, r2 = self.eval_metrics(y_test, prediction)

                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Log the model to MLflow
                if tracking_url_type_store != "file":
                    # Register the model if the tracking store is not a file
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    # Log model without registration if it's stored in a file-based system
                    mlflow.sklearn.log_model(model, "model")

                logging.info("Model evaluation and logging to MLflow completed.")

        except Exception as e:
            logging.error("Exception occurred during model evaluation")
            raise customexception(e, sys)

if __name__ == "__main__":
    # Load the train and test data from CSV files
    train_data_path = 'artifacts/transformed_train.csv'
    test_data_path = 'artifacts/transformed_test.csv'
    
    # Load CSV files using pandas
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # Convert DataFrames to numpy arrays
    train_array = train_data.values
    test_array = test_data.values

    # Create an instance of the ModelEvaluation class and initiate evaluation
    evaluator = ModelEvaluation()
    evaluator.initiate_model_evaluation(train_array, test_array)
