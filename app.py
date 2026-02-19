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
import folium
from streamlit_folium import st_folium
import heapq
import math

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Crime Intelligence System", layout="wide")

st.title("🚨 AI Crime Intelligence + Patrol Routing System")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("crime.data.csv")

df["Case Closed"] = df["Case Closed"].map({"Yes": 1, "No": 0})
df["Hour"] = pd.to_datetime(df["Time of Occurrence"], errors="coerce").dt.hour
df = df.dropna(subset=["Hour"])

df_model = df[["Crime Code", "Victim Age", "Police Deployed", "Hour", "Case Closed"]]

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
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, accuracy = train_model()

st.write(f"🎯 Model Accuracy: {accuracy*100:.2f}%")

# ---------------------------------------------------
# VOICE FUNCTION
# ---------------------------------------------------
def play_voice(message):
    tts = gTTS(text=message, lang='en')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    st.audio(temp_audio.name, autoplay=True)

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
st.header("🔍 Crime Prediction")

col1, col2 = st.columns(2)

with col1:
    crime_code = st.number_input("Crime Code", min_value=1)
    victim_age = st.slider("Victim Age", 1, 100)

with col2:
    police_deployed = st.slider("Police Deployed", 0, 50)
    hour = st.slider("Hour", 0, 23)

if st.button("🚨 Predict Case Outcome"):

    input_data = np.array([[crime_code, victim_age, police_deployed, hour]])
    probability = model.predict_proba(input_data)[0][1] * 100
    prediction = model.predict(input_data)

    st.write(f"Closure Probability: {probability:.2f}%")

    if prediction[0] == 1:
        st.success("Case Likely Closed")
        play_voice("Investigation successful. Case likely to be closed.")
    else:
        st.error("High Risk Crime Detected!")
        play_voice("Emergency alert. Police units dispatched immediately.")

# ---------------------------------------------------
# ANALYTICS
# ---------------------------------------------------
st.header("📊 Crime Analytics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Cases", len(df))
col2.metric("Closed Cases", int(df["Case Closed"].sum()))
col3.metric("Unsolved Cases", int(len(df) - df["Case Closed"].sum()))

st.subheader("Top Cities")
st.bar_chart(df["City"].value_counts().head(10))

# ---------------------------------------------------
# DSA PATROL ROUTING (NO API)
# ---------------------------------------------------
st.header("🗺️ Patrol Routing using Dijkstra Algorithm")

st.write("This routing uses manual graph + Dijkstra algorithm (Pure DSA Implementation).")

# Sample City Graph (Delhi simplified)
graph = {
    "A": {"B": 4, "C": 2},
    "B": {"A": 4, "C": 5, "D": 10},
    "C": {"A": 2, "B": 5, "D": 3},
    "D": {"B": 10, "C": 3}
}

coordinates = {
    "A": (28.6139, 77.2090),
    "B": (28.6200, 77.2100),
    "C": (28.6150, 77.2200),
    "D": (28.6250, 77.2300)
}

def dijkstra(graph, start, end):
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    path = []
    node = end
    while node:
        path.append(node)
        node = previous[node]
    path.reverse()

    return path, distances[end]

source_node = st.selectbox("Select Source", list(graph.keys()))
dest_node = st.selectbox("Select Destination", list(graph.keys()))

if st.button("Find Shortest Path"):

    path, distance = dijkstra(graph, source_node, dest_node)

    st.success(f"Shortest Distance: {distance}")
    st.write("Path:", " → ".join(path))

    m = folium.Map(location=coordinates[source_node], zoom_start=14)

    route_coords = [coordinates[node] for node in path]

    folium.PolyLine(route_coords, color="red", weight=5).add_to(m)

    for node in path:
        folium.Marker(coordinates[node], tooltip=node).add_to(m)

    st_folium(m, width=800, height=500)
