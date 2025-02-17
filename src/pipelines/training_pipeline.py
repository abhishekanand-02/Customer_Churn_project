import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.logger import logging
from src.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

def start_training_pipeline():
    try:
        # Step 1: Data Ingestion
        logging.info("Starting data ingestion...")
        data_ingestion_obj = DataIngestion()
        train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

        # Step 2: Data Transformation
        logging.info("Starting data transformation...")
        data_transformation_obj = DataTransformation()
        train_arr, test_arr = data_transformation_obj.initialize_data_transformation(train_data_path, test_data_path)

        # Step 3: Model Training
        logging.info("Starting model training...")
        model_trainer_obj = ModelTrainer()
        model_trainer_obj.initiate_model_training(train_arr, test_arr) 

        # Step 4: Model Evaluation
        logging.info("Starting model evaluation...")
        model_eval_obj = ModelEvaluation()
        model_eval_obj.initiate_model_evaluation(train_arr, test_arr)

        logging.info("Training pipeline completed successfully!")

    except Exception as e:
        logging.error("Exception occurred in the training pipeline")
        raise customexception(e, sys)

if __name__ == "__main__":
    start_training_pipeline()