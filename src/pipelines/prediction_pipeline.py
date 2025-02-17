import os
import sys
import pandas as pd
from src.exception import customexception 
from src.logger import logging
from src.utils.utils import load_object
from sklearn.preprocessing import LabelEncoder

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing the prediction pipeline object...")

    def predict(self, features: pd.DataFrame, target: pd.Series = None):
        try:
            # Load preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model from the artifacts directory...")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Apply preprocessor transformation and predict
            logging.info("Applying preprocessing to the input data...")
            scaled_features = preprocessor.transform(features)
            predictions = model.predict(scaled_features)

            # Skip model score for single sample predictions
            if target is not None and len(target) > 1:
                score = model.score(scaled_features, target)
                logging.info(f"Model Score: {score}")
            else:
                logging.info("Model score calculation skipped for single sample prediction.")

            # Log actual vs predicted values if target is available
            if target is not None:
                for actual, predicted in zip(target, predictions):
                    logging.info(f"Actual: {actual}, Predicted: {predicted}")
            else:
                for pred in predictions:
                    logging.info(f"Predicted: {pred}")

            logging.info("Prediction process completed successfully.")
            return predictions

        except Exception as e:
            logging.error("An error occurred during prediction.")
            raise customexception(e, sys)


class CustomData:
    def __init__(self, age: float, gender: str, tenure: float, usage_frequency: float,
                 support_calls: int, payment_delay: float, subscription_type: str,
                 contract_length: str, total_spend: float, last_interaction: float):
        
        self.age = age
        self.gender = gender
        self.tenure = tenure
        self.usage_frequency = usage_frequency
        self.support_calls = support_calls
        self.payment_delay = payment_delay
        self.subscription_type = subscription_type
        self.contract_length = contract_length
        self.total_spend = total_spend
        self.last_interaction = last_interaction

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            # Prepare input data dictionary
            data_dict = {
                'Age': [self.age],
                'Gender': [self.gender],
                'Tenure': [self.tenure],
                'Usage Frequency': [self.usage_frequency],
                'Support Calls': [self.support_calls],
                'Payment Delay': [self.payment_delay],
                'Subscription Type': [self.subscription_type],
                'Contract Length': [self.contract_length],
                'Total Spend': [self.total_spend],
                'Last Interaction': [self.last_interaction]
            }

            # Convert dictionary to DataFrame
            df = pd.DataFrame(data_dict)
            logging.info("Dataframe created successfully.")

            # Encode categorical features
            df = self.encode_categorical_columns(df)

            return df
        except Exception as e:
            logging.error("Exception occurred while creating the dataframe.")
            raise customexception(e, sys)

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            label_encoder = LabelEncoder()
            categorical_columns = ['Contract Length', 'Gender', 'Subscription Type']

            for column in categorical_columns:
                if df[column].dtype == 'object':
                    df[column] = label_encoder.fit_transform(df[column].fillna('Unknown'))
                    logging.info(f"Encoded '{column}' column.")

            logging.info(f"Unique values in 'Contract Length': {df['Contract Length'].unique()}")
            return df
        except Exception as e:
            logging.error(f"Error encoding categorical columns: {str(e)}")
            raise customexception(e, sys)


# if __name__ == "__main__":
#     try:
#         # Example feature set
#         features = pd.DataFrame({
#             'Age': [25],
#             'Gender': ['Male'],
#             'Tenure': [2],
#             'Usage Frequency': [3],
#             'Support Calls': [1],
#             'Payment Delay': [0.5],
#             'Subscription Type': ['Basic'],
#             'Contract Length': ['Year'],
#             'Total Spend': [120.0],
#             'Last Interaction': [17.0]
#         })

#         # Create an instance of CustomData
#         custom_data = CustomData(
#             age=25,
#             gender="Male",
#             tenure=2,
#             usage_frequency=3,
#             support_calls=1,
#             payment_delay=0.5,
#             subscription_type="Basic",
#             contract_length="Year",
#             total_spend=120.0,
#             last_interaction=17.0
#         )

#         # Convert custom data to DataFrame
#         data_df = custom_data.get_data_as_dataframe()

#         # Use the prediction pipeline
#         predict_pipeline = PredictPipeline()
#         predictions = predict_pipeline.predict(features=data_df)

#         logging.info(f"Predictions: {predictions}")

#     except Exception as e:
#         logging.error(f"Error in main execution: {str(e)}")
