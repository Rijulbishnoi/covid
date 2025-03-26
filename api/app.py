import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Ensure the correct path)
model_path = "api/model.pkl"  # Correct path inside the container
model = joblib.load(model_path)

# Streamlit UI
st.title("Machine Learning Model Predictor")

# Input fields
city = st.number_input("city", value=0.0,)
gender = st.number_input("gender", value=0.0)
fever = st.number_input("fever", value=0.0)
cough = st.number_input("cough", value=0.0)
age = st.number_input("age", value=0.0)

# Predict button
if st.button("Predict"):
    features = pd.DataFrame([[city, gender, fever, cough, city ]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {int(prediction[0])}")
