import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# Unzip model if not already extracted
if not os.path.exists("crop_yield_model.pkl"):
    with zipfile.ZipFile("crop_yield_model.zip", "r") as zip_ref:
        zip_ref.extractall()

# Load model and encoders
model = joblib.load("crop_yield_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

st.title("🌾 Crop Yield Prediction")
st.write("Predict crop yield based on agricultural and environmental factors.")

st.divider()

# User Inputs
crop = st.selectbox("Select Crop", crop_encoder.classes_)
season = st.selectbox("Select Season", season_encoder.classes_)
state = st.selectbox("Select State", state_encoder.classes_)

year = st.number_input("Crop Year", min_value=1990, max_value=2035, value=2024)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)
fertilizer = st.number_input("Fertilizer usage (kg per hectare)", min_value=0.0)
pesticide = st.number_input("Pesticide usage (kg per hectare)", min_value=0.0)

st.divider()

if st.button("Predict Crop Yield"):

    crop_encoded = crop_encoder.transform([crop])[0]
    season_encoded = season_encoder.transform([season])[0]
    state_encoded = state_encoder.transform([state])[0]

    input_data = pd.DataFrame({
        "Crop": [crop_encoded],
        "Crop_Year": [year],
        "Season": [season_encoded],
        "State": [state_encoded],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide]
    })

    prediction = model.predict(input_data)

    st.success(f"🌱 Predicted Crop Yield: {prediction[0]:.2f} tonnes per hectare")
