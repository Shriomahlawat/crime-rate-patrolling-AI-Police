@st.cache_resource
def train_model():
    df = pd.read_csv("crime.data.csv")

    # Convert Case Closed to numeric
    df["Case Closed"] = df["Case Closed"].map({"Yes": 1, "No": 0})

    # Extract hour from Time of Occurrence
    df["Hour"] = pd.to_datetime(df["Time of Occurrence"]).dt.hour

    # Convert categorical features
    df = pd.get_dummies(df, columns=["City", "Crime Domain", "Weapon Used"], drop_first=True)

    # Features
    X = df.drop(columns=[
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Date Case Closed",
        "Crime Description",
        "Victim Gender",
        "Case Closed"
    ])

    y = df["Case Closed"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, X.columns
model, feature_columns = train_model()

if st.button("Predict Case Closure Probability"):

    input_dict = {}

    for col in feature_columns:
        input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Case likely to be closed quickly")
    else:
        st.error("⚠ Case may remain unsolved")
