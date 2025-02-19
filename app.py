from flask import Flask, render_template, request
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.logger import logging
from datetime import datetime

app = Flask(__name__)

def process_input(form):
    """Processes user input from the form"""
    try:
        age = int(form['age'])
        gender = form['gender']
        tenure = int(form['tenure'])
        usage_frequency = int(form['usage_frequency'])
        support_calls = int(form['support_calls'])
        payment_delay = int(form['payment_delay'])
        subscription_type = form['subscription_type']
        contract_length = form['contract_length']
        total_spend = int(form['total_spend'])
        last_interaction = datetime.strptime(form['last_interaction'], '%Y-%m-%d')
        
        # Calculate days since last interaction
        days_since_last_interaction = (datetime.today().date() - last_interaction.date()).days

        # Prepare data
        data = {
            'age': age,
            'gender': gender,
            'tenure': tenure,
            'usage_frequency': usage_frequency,
            'support_calls': support_calls,
            'payment_delay': payment_delay,
            'subscription_type': subscription_type,
            'contract_length': contract_length,
            'total_spend': total_spend,
            'last_interaction': days_since_last_interaction
        }

        return pd.DataFrame(data, index=[0])
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the home page"""
    if request.method == "POST":
        try:
            print("Received POST request!")  # Debugging
            print(request.form)  # Print form data
            
            user_data = process_input(request.form)
            print(user_data)  # Check processed input
            
            # Encoding categorical variables
            gender_mapping = {'Male': 0, 'Female': 1}
            subscription_mapping = {'Standard': 0, 'Basic': 1, 'Premium': 2}
            contract_mapping = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}

            # Convert to numerical values
            user_data['gender'] = user_data['gender'].map(gender_mapping)
            user_data['subscription_type'] = user_data['subscription_type'].map(subscription_mapping)
            user_data['contract_length'] = user_data['contract_length'].map(contract_mapping)

            # Create CustomData instance
            custom_data = CustomData(**user_data.iloc[0].to_dict())
            input_data = custom_data.get_data_as_dataframe()

            # Run Prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_data)[0]

            print("Prediction:", prediction)  # Debug output

            return render_template("result.html", prediction=prediction)

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template("index.html", error="An error occurred. Try again!")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
