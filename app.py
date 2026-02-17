import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(page_title="AI Crime Patrol System", layout="wide")

# -------------------------
# TRAIN MODEL AUTOMATICALLY
# -------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("dataset.csv")
    df = df.dropna()

    X = df[["latitude", "longitude", "hour", "day"]]
    y = df["crime_occurred"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

model = train_model()

# -------------------------
# UI
# -------------------------

st.title("🚨 AI Crime Prediction & Patrol System")
st.markdown("### Protecting Society with Artificial Intelligence")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/patrol.jpg", use_container_width=True)

with col2:
    st.image("assets/crime_scene.jpg", use_container_width=True)

with col3:
    st.image("assets/murder_alert.jpg", use_container_width=True)

st.markdown("---")

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
