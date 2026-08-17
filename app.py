import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

@st.cache_resource
def load_resources():
    # Load scaler and encoder
    scaler = joblib.load('model/scaler.joblib')
    le = joblib.load('model/label_encoder.joblib')
    
    # Dictionary of available models
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbor": "k-nearest_neighbor.joblib",
        "Naive Bayes (Gaussian)": "naive_bayes_(gaussian).joblib",
        "Random Forest": "random_forest.joblib"
    }
    return scaler, le, model_files

scaler, le, model_files = load_resources()

st.title("🫘 Dry Bean Classification Explorer")

# Sidebar Model Selection
selected_name = st.sidebar.selectbox("Select a Pre-trained Model", list(model_files.keys()))

# Load specific model
model = joblib.load(f"model/{model_files[selected_name]}")

st.info(f"Currently using: **{selected_name}**")

# Input UI
st.subheader("Predict Bean Class")
with st.expander("Input Feature Values", expanded=True):
    cols = st.columns(4)
    # Feature names (standard Dry Bean features)
    features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 
                'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 
                'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
    
    user_inputs = []
    for i, feat in enumerate(features):
        val = cols[i % 4].number_input(feat, value=0.0, format="%.6f")
        user_inputs.append(val)

if st.button("Run Classification"):
    input_arr = np.array(user_inputs).reshape(1, -1)
    
    # Apply scaling for specific models
    if selected_name in ["Logistic Regression", "K-Nearest Neighbor"]:
        input_arr = scaler.transform(input_arr)
        
    prediction = model.predict(input_arr)
    class_name = le.inverse_transform(prediction)[0]
    
    st.success(f"Prediction: **{class_name}**")
