from preprocessing import preprocess
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime


model = joblib.load("model/model.sav")

def main():
    st.title('Customer Churn Prediction App')

    st.markdown("""This app predicts if a customer will churn based on their parameters.
                You can input customer data for prediction here:""")

    add_selectbox = st.sidebar.selectbox("Choose Prediction Mode", ("Online"))

    if add_selectbox == "Online":
        st.info("Input data for prediction below:")

        gender = st.selectbox('Gender:', ('Female', 'Male'), index = 1)
        city = st.selectbox('City:', (
            'city1', 'city2', 'city3', 'city4', 'city5', 'city6', 'city7', 'city8', 'city9', 'city10',
            'city11', 'city12', 'city13', 'city14', 'city15', 'city16', 'city17', 'city18', 'city19', 'city20', 'city21'), index = 4)
        age = st.number_input('Age:', min_value=1, max_value=150, value=36)
        registered_via = st.selectbox('Registered Via:', (
            'Registered Via 3', 'Registered Via 4', 'Registered Via 7', 'Registered Via 9', 'Registered Via 13'), index = 3)
        registration_init_time = st.date_input('Registration date', min_value = datetime(2004, 1, 1), max_value = datetime(2017, 12, 31), value = datetime(2010, 9, 11))
        payment_method_id = st.selectbox('Payment Method ID:', (
            'Payment Method 3', 'Payment Method 6', 'Payment Method 8', 'Payment Method 10', 'Payment Method 11',
            'Payment Method 12', 'Payment Method 13', 'Payment Method 14', 'Payment Method 15', 'Payment Method 16',
            'Payment Method 17', 'Payment Method 18', 'Payment Method 19', 'Payment Method 20', 'Payment Method 21',
            'Payment Method 22', 'Payment Method 23', 'Payment Method 26', 'Payment Method 27', 'Payment Method 28',
            'Payment Method 29', 'Payment Method 30', 'Payment Method 31', 'Payment Method 32', 'Payment Method 33',
            'Payment Method 34', 'Payment Method 35', 'Payment Method 36', 'Payment Method 37', 'Payment Method 38',
            'Payment Method 39', 'Payment Method 40', 'Payment Method 41'), index = 28)
        payment_plan_days = st.number_input('Payment Plan (Days):', min_value=1, max_value=365, value=30)
        plan_list_price = st.number_input('Plan List Price:', min_value=0, max_value=1000, value=149)
        actual_amount_paid = st.number_input('Actual Amount Paid:', min_value=0, max_value=1000, value=149)
        is_auto_renew = st.selectbox('Auto Renew:', ('Yes', 'No'), index = 0)
        transaction_date = st.date_input('Transaction date', min_value = registration_init_time, max_value = datetime(2017, 12, 31), value = datetime(2017, 3, 15))
        membership_expire_date = st.date_input('Membership expire date', min_value = transaction_date, max_value = datetime(2017, 12, 31), value = datetime(2017, 4, 15))
        is_cancel = st.selectbox('Is Cancel:', ('Yes', 'No'), index = 1)
        date = st.date_input('Activity date', min_value = registration_init_time,  max_value = datetime(2017, 12, 31), value = datetime(2017, 3, 25))
        num_25 = st.number_input('Num 25:', min_value = 0, value = 1)
        num_50 = st.number_input('Num 50:', min_value = 0, value = 0)
        num_75 = st.number_input('Num 75:', min_value = 0, value=2)
        num_985 = st.number_input('Num 985:', min_value = 0, value = 1)
        num_100 = st.number_input('Num 100:', min_value = 0, value = 6)
        num_unq = st.number_input('Num Unq:', min_value = 0, value = 10)
        total_secs = st.number_input('Total Seconds:', min_value = 0, value = 2000)

        data = {
            'gender': gender,
            'city': city,
            'bd': age,
            'registered_via': registered_via,
            'registration_init_time': registration_init_time,
            'payment_method_id': payment_method_id,
            'payment_plan_days': payment_plan_days,
            'plan_list_price': plan_list_price,
            'actual_amount_paid': actual_amount_paid,
            'is_auto_renew': is_auto_renew,
            'transaction_date': transaction_date,
            'membership_expire_date': membership_expire_date,
            'is_cancel': is_cancel,
            'date': date,
            'num_25': num_25,
            'num_50': num_50,
            'num_75': num_75,
            'num_985': num_985,
            'num_100': num_100,
            'num_unq': num_unq,
            'total_secs': total_secs
        }

        features_df = pd.DataFrame.from_dict([data])

        st.write('Overview of input data:')
        st.dataframe(features_df)

        processed_data = preprocess(features_df)
        prediction = model.predict(processed_data)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with the services.')
        

if __name__ == '__main__':
    main()