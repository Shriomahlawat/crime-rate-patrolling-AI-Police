import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gtts import gTTS
import tempfile
import base64

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Crime Intelligence System", layout="wide")

# ---------------------------------------------------
# DARK THEME
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
    color:#ff2c2c;
}
.section-title {
    font-size:26px;
    color:#ff4d4d;
}
@keyframes flash {
  0% {background-color:#0f0f0f;}
  50% {background-color:#8b0000;}
  100% {background-color:#0f0f0f;}
}
.flash {
  animation: flash 1s infinite;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚨 AI Crime Intelligence & Prediction System</p>', unsafe_allow_html=True)
st.markdown("### Advanced Crime Analytics & Case Closure Prediction")

# ---------------------------------------------------
# HERO IMAGES
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

for img, col in zip(["patrol.jpg", "crime_scene.jpg", "murder_alert.jpg"], [col1, col2, col3]):
    with col:
        if os.path.exists(img):
            st.image(img, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
if not os.path.exists("crime.data.csv"):
    st.error("crime.data.csv not found!")
    st.stop()

df = pd.read_csv("crime.data.csv")

df["Case Closed"] = df["Case Closed"].map({"Yes": 1, "No": 0})

df["Hour"] = pd.to_datetime(
    df["Time of Occurrence"],
    dayfirst=True,
    errors="coerce"
).dt.hour

df = df.dropna(subset=["Hour"])

df_model = df[[
    "Crime Code",
    "Victim Age",
    "Police Deployed",
    "Hour",
    "Case Closed"
]].dropna()

X = df_model.drop("Case Closed", axis=1)
y = df_model["Case Closed"]

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

st.write(f"### 🎯 Model Accuracy: {accuracy*100:.2f}%")

st.markdown("---")

# ---------------------------------------------------
# CINEMATIC ALERT FUNCTION
# ---------------------------------------------------
def play_cinematic_alert(message):

    # 🔴 Flash Screen
    st.markdown('<div class="flash"></div>', unsafe_allow_html=True)

    # 🚔 Siren Sound
    siren_url = "https://www.soundjay.com/misc/sounds/police-siren-01.mp3"

    # 🎵 Crime Intro Theme
    intro_url = "https://www.soundjay.com/button/beep-07.mp3"

    # 🔊 Generate Voice
    tts = gTTS(text=message, lang='en')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    with open(temp_audio.name, "rb") as f:
        audio_bytes = f.read()
        voice_base64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay>
        <source src="{siren_url}" type="audio/mp3">
    </audio>

    <audio autoplay>
        <source src="{intro_url}" type="audio/mp3">
    </audio>

    <audio autoplay>
        <source src="data:audio/mp3;base64,{voice_base64}" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
st.markdown('<p class="section-title">🔍 Case Closure Prediction</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    crime_code = st.number_input("Crime Code", min_value=1)
    victim_age = st.slider("Victim Age", 1, 100)

with col2:
    police_deployed = st.slider("Police Deployed", 0, 50)
    hour = st.slider("Hour of Crime", 0, 23)

if st.button("🚨 Predict Case Outcome"):

    input_data = np.array([[crime_code, victim_age, police_deployed, hour]])

    probability = model.predict_proba(input_data)[0][1] * 100
    prediction = model.predict(input_data)

    st.write(f"### 🔎 Closure Probability: {probability:.2f}%")

    if prediction[0] == 1:
        st.success("✅ Case Likely to be Closed")
        play_cinematic_alert(
            "Investigation update. The case is likely to be successfully closed. Police department is in control."
        )
    else:
        st.error("⚠ High Risk Crime Detected!")
        play_cinematic_alert(
            "Crime alert activated. High risk detected. Police units are being dispatched immediately."
        )

st.markdown("---")

st.subheader("📊 Crime Analytics Dashboard")
col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", len(df))
col2.metric("Closed Cases", int(df["Case Closed"].sum()))
col3.metric("Unsolved Cases", int(len(df) - df["Case Closed"].sum()))

st.markdown("🚀 Built by Shriom | Cinematic Crime Intelligence System")
