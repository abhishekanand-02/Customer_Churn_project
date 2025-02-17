import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customexception
import os
import sys
from src.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LogisticRegression  

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data.')

            # Splitting the features (X) and target (y) from the training and testing arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]    
            )

            models = {
                'LogisticRegression': LogisticRegression()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            logging.info(f"Model Performance Report:\n{model_report}")
            print(f"Model Performance Report:\n{model_report}")

            logging.info('\n====================================================================================')

            # As there is only one model, it's always the best model
            best_model_score = list(model_report.values())[0] 
            best_model_name = 'LogisticRegression'

            logging.info(f"Best Model Found: Model Name: {best_model_name}, Accuracy: {best_model_score}")
            # print(f"Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_score}")
            logging.info('\n====================================================================================')

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    train_data_path = 'artifacts/processed_train.csv'
    test_data_path = 'artifacts/processed_test.csv'
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    train_array = train_data.values
    test_array = test_data.values

    trainer = ModelTrainer()
    trainer.initiate_model_training(train_array, test_array)
