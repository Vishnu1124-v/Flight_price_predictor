import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Load model from file next to this script
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")

# Helper to download model from a URL
def _download_model(url, dest_path, timeout=30):
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        if resp.status_code == 200:
            with open(dest_path, "wb") as fo:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fo.write(chunk)
            return True
    except Exception:
        return False
    return False

# Attempt to load model; if missing, try to download from MODEL_URL or GitHub raw
DEFAULT_MODEL_URL = "https://raw.githubusercontent.com/Vishnu1124-v/Flight_price_predictor/main/best_model.pkl"
model = None
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # try download from env var or default GitHub raw URL
    model_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)
    downloaded = False
    if model_url:
        downloaded = _download_model(model_url, model_path)

    if downloaded:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.set_page_config(page_title="Flight Price Predictor", layout="centered")
            st.title("Flight Price Prediction App")
            st.error(f"Failed to load downloaded model: {e}")
            st.markdown("Host the trained `best_model.pkl` in your repository or set the `MODEL_URL` app environment variable to a reachable raw file URL.")
            st.stop()
    else:
        st.set_page_config(page_title="Flight Price Predictor", layout="centered")
        st.title("Flight Price Prediction App")
        st.error("Missing model file: best_model.pkl not found in the app folder, and automatic download failed.")
        st.markdown("Please add `best_model.pkl` to the project root and redeploy, or host it at a stable URL and set the `MODEL_URL` environment variable for your deployment.")
        st.stop()

# Suppress sklearn version mismatch warning coming from unpickling
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(page_title="Flight Price Predictor", layout="centered")

st.title("✈️ Flight Price Prediction App")
st.write("Predict flight ticket prices using Machine Learning")

st.divider()

# -------- INPUTS -------- #
airline = st.selectbox(
    "Airline",
    ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara", "GoAir"]
)

source = st.selectbox(
    "Source City",
    ["Delhi", "Mumbai", "Chennai", "Kolkata"]
)

destination = st.selectbox(
    "Destination City",
    ["Cochin", "Delhi", "Mumbai", "Hyderabad", "Kolkata"]
)

total_stops = st.selectbox(
    "Total Stops",
    [0, 1, 2, 3]
)

journey_day = st.number_input("Journey Day", 1, 31, 1)
journey_month = st.number_input("Journey Month", 1, 12, 1)

duration_hours = st.number_input("Duration (Hours)", 0, 50, 2)
duration_minutes = st.number_input("Duration (Minutes)", 0, 59, 30)

# Departure time (optional — used to compute arrival)
dep_hour = st.number_input("Departure Hour (0-23)", 0, 23, 9)
dep_minute = st.number_input("Departure Minute (0-59)", 0, 59, 0)

st.divider()

# -------- ENCODING -------- #
# NOTE: Encoding must match your notebook logic

airline_map = {
    "IndiGo": 0,
    "Air India": 1,
    "Jet Airways": 2,
    "SpiceJet": 3,
    "Vistara": 4,
    "GoAir": 5
}

source_map = {
    "Delhi": 0,
    "Mumbai": 1,
    "Chennai": 2,
    "Kolkata": 3
}

destination_map = {
    "Cochin": 0,
    "Delhi": 1,
    "Mumbai": 2,
    "Hyderabad": 3,
    "Kolkata": 4
}

# Convert to numeric
airline = airline_map[airline]
source = source_map[source]
destination = destination_map[destination]

# -------- PREDICTION -------- #
if st.button("💰 Predict Price"):
    # Build full feature vector expected by the model (20 features)
    # Model.feature_names_in_ = ['Airline','Source','Destination','Total_Stops',
    # 'Additional_Info','Day','Month','Year','Dep_Hour','Dep_Minute',
    # 'Arrival_hour','Arrival_minute','Arrival_month','Duration_hours',
    # 'Duration_minutes','Route_1','Route_2','Route_3','Route_4','Route_5']

    # Reasonable defaults / simple derivations
    additional_info = 0  # encoded 'No info' -> 0 (matching training encoding assumption)
    year = 2019

    # Compute arrival time from departure + duration
    total_minutes = dep_hour * 60 + dep_minute + int(duration_hours) * 60 + int(duration_minutes)
    arrival_hour = (total_minutes // 60) % 24
    arrival_minute = total_minutes % 60
    arrival_month = journey_month

    # Simple route encoding: mark first (total_stops+1) route legs as 1
    routes = [0, 0, 0, 0, 0]
    legs = max(1, int(total_stops) + 1)
    for i in range(min(5, legs)):
        routes[i] = 1


    # Build a named DataFrame matching the model's trained feature names
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
    else:
        cols = ['Airline','Source','Destination','Total_Stops','Additional_Info','Day','Month','Year','Dep_Hour','Dep_Minute','Arrival_hour','Arrival_minute','Arrival_month','Duration_hours','Duration_minutes','Route_1','Route_2','Route_3','Route_4','Route_5']

    row = {
        'Airline': int(airline),
        'Source': int(source),
        'Destination': int(destination),
        'Total_Stops': int(total_stops),
        'Additional_Info': int(additional_info),
        'Day': int(journey_day),
        'Month': int(journey_month),
        'Year': int(year),
        'Dep_Hour': int(dep_hour),
        'Dep_Minute': int(dep_minute),
        'Arrival_hour': int(arrival_hour),
        'Arrival_minute': int(arrival_minute),
        'Arrival_month': int(arrival_month),
        'Duration_hours': int(duration_hours),
        'Duration_minutes': int(duration_minutes),
        'Route_1': int(routes[0]),
        'Route_2': int(routes[1]),
        'Route_3': int(routes[2]),
        'Route_4': int(routes[3]),
        'Route_5': int(routes[4]),
    }

    df = pd.DataFrame([row], columns=cols)

    prediction = model.predict(df)[0]

    st.success(f"Estimated Flight Price: ₹ {int(prediction):,}")


# Footer
st.write("Done by Vishnu1124")