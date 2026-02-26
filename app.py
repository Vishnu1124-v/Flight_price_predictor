import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

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

    features = np.array([[
        airline,
        source,
        destination,
        int(total_stops),
        additional_info,
        int(journey_day),
        int(journey_month),
        int(year),
        int(dep_hour),
        int(dep_minute),
        int(arrival_hour),
        int(arrival_minute),
        int(arrival_month),
        int(duration_hours),
        int(duration_minutes),
        routes[0],
        routes[1],
        routes[2],
        routes[3],
        routes[4],
    ]])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Flight Price: ₹ {int(prediction):,}")


# Footer
st.write("Done by Vishnu1124")