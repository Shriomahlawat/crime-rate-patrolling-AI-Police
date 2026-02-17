import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Crime Predictor", layout="wide")

# ---------------------------------------------------
# PREMIUM DARK UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: white;
}
.big-title {
    font-size:40px;
    font-weight:bold;
    color:red;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚨 AI Crime Prediction & Patrol System</p>', unsafe_allow_html=True)
st.markdown("### Intelligent Crime Analysis & Case Closure Prediction")

# ---------------------------------------------------
# FRONT PAGE IMAGES
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.image("assets/patrol.jpg", use_container_width=True)

with col2:
    st.image("assets/crime_scene.jpg", use_container_width=True)

with col3:
    st.image("assets/murder_alert.jpg", use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET SAFELY
# ---------------------------------------------------
if not os.path.exists("crime.data.csv"):
    st.error("crime.data.csv not found in repository!")
    st.stop()

try:
    df = pd.read_csv("crime.data.csv")
except:
    st.error("Error reading crime.data.csv. Check file format.")
    st.stop()

if df.empty:
    st.error("Dataset is empty!")
    st.stop()

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
df["Case Closed"] = df["Case Closed"].map({"Yes": 1, "No": 0})

df["Hour"] = pd.to_datetime(df["Time of Occurrence"]).dt.hour

# Drop unnecessary columns
df_model = df[[
    "Crime Code",
    "Victim Age",
    "Police Deployed",
    "Hour",
    "Case Closed"
]]

df_model = df_model.dropna()

X = df_model.drop("Case Closed", axis=1)
y = df_model["Case Closed"]

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------------------------------------
# PREDICTION UI
# ---------------------------------------------------
st.header("🔍 Predict Case Closure Probability")

crime_code = st.number_input("Crime Code", min_value=1)
victim_age = st.slider("Victim Age", 1, 100)
police_deployed = st.slider("Police Deployed", 0, 50)
hour = st.slider("Hour of Crime", 0, 23)

if st.button("Predict Case Status"):

    input_data = np.array([[crime_code, victim_age, police_deployed, hour]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ This Case is Likely to be Closed")
    else:
        st.error("⚠ High Risk of Case Remaining Unsolved")

st.markdown("---")

# ---------------------------------------------------
# ANALYTICS SECTION
# ---------------------------------------------------
st.header("📊 Crime Analytics Dashboard")

total_cases = len(df)
closed_cases = df["Case Closed"].sum()
unsolved = total_cases - closed_cases

st.metric("Total Cases", total_cases)
st.metric("Closed Cases", int(closed_cases))
st.metric("Unsolved Cases", int(unsolved))

st.markdown("Built by Shriom | AI Crime Prediction Project 🚀")
