import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model
model = joblib.load("crime_model.pkl")

# Page Config
st.set_page_config(page_title="AI Crime Patrol System", layout="wide")

# ---------- HEADER ----------
st.title("🚨 AI Crime Prediction & Patrol System")
st.markdown("### Protecting Society with Artificial Intelligence")

# ---------- FRONT IMAGES ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/patrol.jpg", caption="Police Patrol Robot", use_column_width=True)

with col2:
    st.image("assets/crime_scene.jpg", caption="Crime Scene Monitoring", use_column_width=True)

with col3:
    st.image("assets/murder_alert.jpg", caption="Murder Alert Detection", use_column_width=True)

st.markdown("---")

# ---------- PREDICTION SECTION ----------
st.header("🔍 Predict Crime Risk")

lat = st.number_input("Latitude")
lon = st.number_input("Longitude")
hour = st.slider("Hour of Day", 0, 23)
day = st.slider("Day of Week (0=Sun,6=Sat)", 0, 6)

if st.button("Predict Crime Risk"):
    input_data = np.array([[lat, lon, hour, day]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ HIGH CRIME RISK AREA!")
    else:
        st.success("✅ Low Risk Area")

st.markdown("---")

st.write("Built by Shriom | AI + DSA + Robotics Project")
