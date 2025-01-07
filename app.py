import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('age.pkl')

def predict_life_expectancy(country, age, cigarettes_per_day):
    input_df = pd.DataFrame([[country, age, cigarettes_per_day]], columns=['Country', 'Age', 'Cigarettes_Per_Day'])
    prediction = model.predict(input_df)[0]
    return prediction

# Set up the webpage
st.title('Life Expectancy Prediction')
st.write('Enter your details to predict life expectancy.')

# Create inputs on the sidebar
country = st.sidebar.text_input("Country", "USA")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
cigarettes_per_day = st.sidebar.number_input("Cigarettes Per Day", min_value=0, max_value=50, value=0)

# Predict button
if st.sidebar.button('Predict Life Expectancy'):
    result = predict_life_expectancy(country, age, cigarettes_per_day)
    st.success(f'The predicted life expectancy is {result:.2f} years.')

