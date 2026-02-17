import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("crime.data.csv")

# Clean data
df = df.dropna()

# Change column names based on YOUR dataset
features = df[["latitude", "longitude", "hour", "day"]]
target = df["crime_occurred"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crime_model.pkl")

print("Model trained & saved successfully!")
