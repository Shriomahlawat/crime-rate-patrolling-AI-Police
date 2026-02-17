import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Crime Intelligence System", layout="wide")

# ---------------------------------------------------
# PREMIUM DARK THEME
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
    font-size:28px;
    color:#ff4d4d;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚨 AI Crime Intelligence & Prediction System</p>', unsafe_allow_html=True)
st.markdown("### Advanced Crime Analytics & Case Closure Prediction")

# ---------------------------------------------------
# HERO IMAGES (ROOT FOLDER)
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists("patrol.jpg"):
        st.image("patrol.jpg", use_container_width=True)

with col2:
    if os.path.exists("crime_scene.jpg"):
        st.image("crime_scene.jpg", use_container_width=True)

with col3:
    if os.path.exists("murder_alert.jpg"):
        st.image("murder_alert.jpg", use_container_width=True)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
if not os.path.exists("crime.data.csv"):
    st.error("crime.data.csv not found!")
    st.stop()

df = pd.read_csv("crime.data.csv")

if df.empty:
    st.error("Dataset is empty!")
    st.stop()

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
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
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)
    return model

model = train_model()

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

if st.button("Predict Case Outcome"):

    input_data = np.array([[crime_code, victim_age, police_deployed, hour]])

    probability = model.predict_proba(input_data)[0][1] * 100
    prediction = model.predict(input_data)

    st.write(f"### 🔎 Closure Probability: {probability:.2f}%")

    if prediction[0] == 1:
        st.success("✅ Case Likely to be Closed")
    else:
        st.error("⚠ Case May Remain Unsolved")

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
# CITY CRIME DISTRIBUTION
# ---------------------------------------------------
st.subheader("🏙 Top 10 Cities by Crime Count")

city_counts = df["City"].value_counts().head(10)
st.bar_chart(city_counts)

# ---------------------------------------------------
# CRIME DOMAIN DISTRIBUTION
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
st.subheader("📈 Feature Importance (Model Insight)")

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.markdown("---")
st.markdown("🚀 Built by Shriom | AI Crime Intelligence System")
