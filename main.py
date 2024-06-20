# Importing all the necessary libries
import numpy as np
import streamlit as st 
import pandas as pd 
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# Load the  model and scaler

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Title of the app
st.title("Placement Prediction App")

# Input fields for user
cgpa = st.number_input("Enter your CGPA", min_value=0, max_value=10, step=1)
iq = st.number_input("Enter your IQ", min_value=0, max_value=200, step=1)

# Button to predict
if st.button("Predict"):
    # Prepare the input data in the format the model expects
    
    input_data = np.array([[iq,cgpa]])
    inputt = scaler.transform(input_data)
    print(input_data)
    print(inputt)
    print(cgpa,iq)
    # Make the prediction
    prediction = model.predict(inputt)

    print(prediction)
    
    # Display the result
    if prediction[0] == 1:
        st.success("You are likely to get placed!")
    else:
        st.error("You are not likely to get placed.")