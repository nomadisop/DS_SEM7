import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np

# Load data and model
data = pd.read_csv('cleaned_featured_books_dataset.csv')
try:
    model = joblib.load('model.pkl')
except Exception as e:
    model = None
    st.warning(f"Model file not found or could not be loaded: {e}")

st.title('Book Prediction Dashboard')
st.write('Explore predictions, SHAP explanations, and data drift checks.')

# Show data preview
st.subheader('Data Preview')
st.dataframe(data.head())

# Make predictions
if model:
    X = data.drop(['target'], axis=1, errors='ignore')
    preds = model.predict(X)
    st.subheader('Predictions')
    st.write(preds)

    # SHAP explanations
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    st.subheader('SHAP Summary Plot')
    st.pyplot(shap.summary_plot(shap_values, X, show=False))

    # Metrics
    st.subheader('Model Metrics')
    # Add your metrics calculation here
    st.write('Accuracy, F1, etc.')

    # Drift check (simple example)
    st.subheader('Data Drift Check')
    st.write('Feature means:', X.mean().to_dict())
else:
    st.warning('Model file not found. Please train and save your model as model.pkl.')
