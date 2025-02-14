import os
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.exception import customexception
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        
        try:
            # Path to the raw data
            data_path = os.path.join("data", "customer_churn_dataset-training-master.csv") 
            logging.info(f"Reading the dataset from {data_path}")
            
            # Check if the data file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"The file {data_path} does not exist.")
            
            # Load data into DataFrame
            data = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully with shape {data.shape}")

            # Log the first few rows of the data for reference
            logging.info(f"First few rows of the data: \n{data.head()}")

            # Make directories for raw data if not exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Perform train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed")

            # Save train and test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")
            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except FileNotFoundError as fnf_error:
            logging.error(f"FileNotFoundError occurred: {fnf_error}")
            raise customexception(fnf_error)

        except Exception as e:
            logging.exception("Exception occurred during data ingestion")
            raise customexception(e)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
