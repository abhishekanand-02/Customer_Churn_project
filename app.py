import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.logger import logging
from datetime import datetime

def user_input_features():
    """Collects user input for prediction"""
    st.sidebar.header('User Input Parameters')

    age = st.sidebar.slider('**Age**', 18, 70, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    tenure = st.sidebar.slider('Tenure (Years)', 1, 60, 2)
    usage_frequency = st.sidebar.slider('Usage Frequency (per month)', 1, 30, 10)
    support_calls = st.sidebar.slider('Support Calls', 0, 10, 2)
    payment_delay = st.sidebar.slider('Payment Delay (in months)', 0, 30, 1)
    subscription_type = st.sidebar.selectbox('Subscription Type', ['Standard', 'Basic', 'Premium'])
    contract_length = st.sidebar.selectbox('Contract Length', ['Monthly', 'Quarterly', 'Annual'])
    total_spend = st.sidebar.slider('Total Spend (INR)', 100, 1000, 500)
    last_interaction = st.sidebar.date_input('Last Interaction Date', max_value=datetime.today().date())

    # Calculate the number of days since last interaction
    days_since_last_interaction = (datetime.today().date() - last_interaction).days

    # Prepare data as a dictionary
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

def display_details():
    """Display detailed description of the inputs."""
    st.markdown("""
    ### Column Descriptions
    - **Age**: The age of the customer (18-70 years).
    - **Gender**: Gender of the customer (Male/Female).
    - **Tenure (Years)**: Number of years the customer has been with the service.
    - **Usage Frequency (per month)**: How often the customer uses the service per month.
    - **Support Calls**: Number of times the customer has called customer support.
    - **Payment Delay (in months)**: How many months the customer has delayed payment.
    - **Subscription Type**: Type of subscription the customer has (Standard, Basic, Premium).
    - **Contract Length**: The length of the customer's contract (Monthly, Quarterly, Annual).
    - **Total Spend**: Total spend by the customer (in INR).
    - **Last Interaction Date**: The number of days since the customer's last interaction.
    """)

def main():
    """Main function to run the Streamlit app"""
    st.title('Customer Churn Prediction App 🛑')
    st.markdown('### Predict whether a customer will churn or not 🚶‍♂️🚶‍♀️')

    # Get user input data
    user_data = user_input_features()

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

    # Initialize the prediction pipeline
    predict_pipeline = PredictPipeline()

    # Layout for buttons in the same row
    col1, col2 = st.columns(2)

    # Predict button
    with col1:
        if st.button('Predict 🔮'):
            try:
                st.write('### Input Data:')
                st.write(input_data)

                # Make the prediction
                prediction = predict_pipeline.predict(input_data)

                # Display prediction result
                st.write('### Prediction Result:')
                if prediction[0] == 0:
                    st.success(f"Customer will continue to use services. The customer is **NOT** likely to churn. Predicted value: {prediction[0]}. 😊")
                else:
                    st.error(f"Customer may unsubscribe. The customer is **likely** to churn. Predicted value: {prediction[0]}. ⚠️")
            except Exception as e:
                logging.error(f"Error during prediction: {str(e)}")
                st.error("An error occurred during prediction. Please try again later.")

    # Show Details button
    with col2:
        if st.button('Show Details ℹ️'):
            display_details()

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    st.markdown("""
    <footer style="text-align:center; padding:20px; font-size:14px; background-color:#f0f0f0;">
        <p><strong>About this App</strong></p>
        <p>This app predicts customer churn based on various customer attributes.</p>
        <p>Churn refers to customers leaving or unsubscribing from the service. Built with Streamlit and machine learning. 🚀</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
