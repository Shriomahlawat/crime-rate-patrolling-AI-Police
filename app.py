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
import requests

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
# AUDIO FUNCTION
# ---------------------------------------------------
def play_police_alert(message):

    # Download siren sound
    siren_url = "https://www.soundjay.com/misc/sounds/police-siren-01.mp3"
    siren_response = requests.get(siren_url)
    siren_audio = siren_response.content

    # Generate voice
    tts = gTTS(text=message, lang='en')
    temp_voice = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_voice.name)

    with open(temp_voice.name, "rb") as f:
        voice_audio = f.read()

    st.audio(siren_audio, format="audio/mp3", autoplay=True)
    st.audio(voice_audio, format="audio/mp3", autoplay=True)

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
        message = "Police update. Investigation under control. The case is likely to be closed successfully."
    else:
        st.error("⚠ High Risk Crime Detected!")
        message = "Emergency alert. High risk crime detected. Police units are on the way immediately."

    play_police_alert(message)

st.markdown("---")

# ---------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------
st.markdown('<p class="section-title">📊 Crime Analytics Dashboard</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

total_cases = len(df)
closed_cases = df["Case Closed"].sum()
unsolved_cases = total_cases - closed_cases

col1.metric("Total Cases", total_cases)
col2.metric("Closed Cases", int(closed_cases))
col3.metric("Unsolved Cases", int(unsolved_cases))

st.markdown("---")

# ---------------------------------------------------
# CITY DISTRIBUTION
# ---------------------------------------------------
st.subheader("🏙 Top 10 Cities by Crime Count")
city_counts = df["City"].value_counts().head(10)
st.bar_chart(city_counts)

# ---------------------------------------------------
# CRIME DOMAIN PIE CHART
# ---------------------------------------------------
st.subheader("🧠 Crime Domain Distribution")
domain_counts = df["Crime Domain"].value_counts().head(5)

fig, ax = plt.subplots()
ax.pie(domain_counts, labels=domain_counts.index, autopct='%1.1f%%')
ax.set_title("Top Crime Domains")
st.pyplot(fig)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------
st.subheader("📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.markdown("---")
st.markdown("🚀 Built by Shriom | AI Crime Intelligence System")
