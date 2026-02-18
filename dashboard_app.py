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
    drop_cols = ['gross_sales', 'book_name', 'author', 'genre', 'publisher', 'language_code']
    features = [col for col in data.columns if col not in drop_cols]
    X = data[features].copy()

    # Encode categorical columns as in training
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].replace('unknown', np.nan)
            X[col] = X[col].fillna('missing')
            X[col] = X[col].astype('category').cat.codes

    # Impute any remaining NaNs with zero
    X = X.fillna(0)

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
    st.write('RMSE and R2 are shown during training. Add more metrics if needed.')

    # Drift check (simple example)
    st.subheader('Data Drift Check')
    st.write('Feature means:', X.mean().to_dict())
else:
    st.warning('Model file not found. Please train and save your model as model.pkl.')
