import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Title
st.title("🚖 PragyanAI Taxi Fare Prediction App (End-to-End ML)")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("taxis.csv")
    df = df[['distance', 'fare']].dropna()

    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['fare'] = pd.to_numeric(df['fare'], errors='coerce')

    return df.dropna()

df = load_data()

# Dataset Preview
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Features & Target
X = df[['distance']]
y = df['fare']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Evaluation Metrics")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# User Input
st.header("🧾 Enter Trip Details")

distance = st.number_input("Step 1: Enter Distance (km)", min_value=0.0, value=5.0)
passengers = st.number_input("Step 2: Number of Passengers", min_value=1, value=1)
hour = st.number_input("Step 3: Enter Hour of the Day", min_value=0, max_value=23, value=12)

# Prediction
if st.button("🚀 Predict Fare"):
    input_data = np.array([[distance]])  # FIXED SHAPE
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Fare: ${prediction[0]:.2f}")
