import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.utils.utils import save_object
from src.logger import logging
from src.exception import customexception

class ModelTrainer:
    def __init__(self):
        logging.info("Model training initiated")

    def train_and_save(self, train_data, test_data):
        try:
            # Split the train data into features and target
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # Define the models to be trained
            models = {
                "Logistic Regression": LogisticRegression(solver='liblinear')
            }

            # Train the model and save the best one (can be based on any metric such as accuracy)
            best_model_name = None
            best_model = None

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, model.predict(X_test))

                if best_model is None or accuracy > best_model_accuracy:
                    best_model = model
                    best_model_name = model_name
                    best_model_accuracy = accuracy

                logging.info(f"Trained {model_name} with accuracy: {accuracy}")

            # Save the best model
            save_object(file_path="artifacts/best_model.pkl", obj=best_model)
            logging.info(f"Best model saved: {best_model_name} with accuracy: {best_model_accuracy}")

            return best_model_name, best_model_accuracy

        except Exception as e:
            logging.error(f"Error occurred during model training: {str(e)}")
            raise customexception(e, sys)

    def run(self, train_data, test_data):
        try:
            # Train and save the model
            model_name, model_accuracy = self.train_and_save(train_data, test_data)
            logging.info(f"Model training completed. Best model: {model_name} with accuracy: {model_accuracy}")
            return model_name, model_accuracy
        except Exception as e:
            logging.error(f"Error occurred in the main block: {str(e)}")
            raise customexception(e, sys)

# Running the model trainer
if __name__ == "__main__":
    # Load the training and test data
    train_data_path = 'artifacts/transformed_train.csv'
    test_data_path = 'artifacts/transformed_test.csv'

    # Load CSV files
    train_data = pd.read_csv(train_data_path).values
    test_data = pd.read_csv(test_data_path).values

    # Create an instance of the ModelTrainer class and run the training
    trainer = ModelTrainer()
    trainer.run(train_data, test_data)
