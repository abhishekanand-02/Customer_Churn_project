import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from src.utils.utils import load_object
from urllib.parse import urlparse
from src.logger import logging
from src.exception import customexception
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, actual, pred):
        """Evaluate the model performance with Accuracy, Precision, Recall, F1, and ROC AUC"""
        try:
            accuracy = accuracy_score(actual, pred)  # Accuracy Score
            precision = precision_score(actual, pred)  # Precision Score
            recall = recall_score(actual, pred)  # Recall Score
            f1 = f1_score(actual, pred)  # F1 Score
            roc_auc = roc_auc_score(actual, pred)  # ROC AUC Score
            logging.info("Evaluation metrics captured: Accuracy, Precision, Recall, F1, ROC AUC")
            return accuracy, precision, recall, f1, roc_auc
        except Exception as e:
            logging.error("Error in calculating evaluation metrics")
            raise customexception(e, sys)

    def initiate_model_evaluation(self, train_array, test_array):
        """Initiate model evaluation and log metrics using MLflow"""
        try:
            # Splitting features and target from the test array
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("artifacts", "model.pkl")

            try:
                model = load_object(model_path)
                logging.info("Model loaded successfully using pickle")
            except Exception as e:
                logging.error("Error occurred while loading the model")
                raise customexception(e, sys)

            logging.info("Model loaded and evaluation initiated")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"Tracking URL type: {tracking_url_type_store}")

            with mlflow.start_run():
                prediction = model.predict(X_test)

                accuracy, precision, recall, f1, roc_auc = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="churn_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logging.info("Model evaluation and logging to MLflow completed.")

        except Exception as e:
            logging.error("Exception occurred during model evaluation")
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        train_data_path = 'artifacts/processed_train.csv'
        test_data_path = 'artifacts/processed_test.csv'
        
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        
        train_array = train_data.values
        test_array = test_data.values

        evaluator = ModelEvaluation()
        evaluator.initiate_model_evaluation(train_array, test_array)

    except Exception as e:
        logging.error("Error occurred during model evaluation execution")
        raise customexception(e, sys)
