import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('marks_model.pkl')

# Title
st.title("🎓 Student Marks Prediction App")

# Input
hours = st.number_input("Enter Study Hours", min_value=0.0, max_value=24.0, step=0.5)

# Button
if st.button("Predict Marks"):
    marks_pred = model.predict(np.array([[hours]]))[0]
    st.success(f"📈 Predicted Marks: {marks_pred:.2f}")
