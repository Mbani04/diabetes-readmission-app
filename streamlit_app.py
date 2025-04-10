
import streamlit as st
import numpy as np
import tensorflow as tf
from utils import preprocess_input

# Load model
model = tf.keras.models.load_model("diabetes_readmission_model.h5")

st.title("🏥 Diabetes Readmission Predictor")
st.write("Predict the risk of 30-day hospital readmission for diabetic patients.")

st.sidebar.header("📋 Patient Information")

age = st.sidebar.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
race = st.sidebar.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
num_medications = st.sidebar.slider("Number of Medications", 1, 50, 10)
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 132, 44)
admission_type = st.sidebar.selectbox("Admission Type", ['Emergency', 'Urgent', 'Elective', 'Other'])
discharge_disposition = st.sidebar.selectbox("Discharge Disposition", ['Discharged to Home', 'Other'])

if st.button("Predict Readmission"):
    input_data = {
        'age': age,
        'gender': gender,
        'race': race,
        'num_medications': num_medications,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'admission_type': admission_type,
        'discharge_disposition': discharge_disposition
    }

    X = preprocess_input(input_data)
    prediction = model.predict(X)[0][0]
    label = "🔴 High Risk of Readmission" if prediction > 0.5 else "🟢 Low Risk of Readmission"
    st.subheader("Prediction Result")
    st.markdown(f"**{label}**")
    st.progress(int(prediction * 100))
