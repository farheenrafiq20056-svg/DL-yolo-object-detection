import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Machine Learning Project #2 - Titanic Survival Predictor
# Beginner friendly version - easy to read and edit
#
# This is a simple CLASSIFICATION model (predicts a category:
# "survived" or "did not survive"), different from Project #1
# which was a REGRESSION model (predicting a number - house price).
# ---------------------------------------------------

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "A simple machine learning model that predicts whether a passenger "
    "would have survived the Titanic, based on their details."
)


# ---------------------------
# Step 1: Load and prepare the data (cached so it only runs once)
# ---------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("Titanic-dataset.csv")

    # Keep only the simple columns we need
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

    # Fill missing ages with the average age (simple approach)
    df["Age"] = df["Age"].fillna(df["Age"].mean())

    # Convert Sex from text to numbers: male = 0, female = 1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    return df


df = load_and_prepare_data()

# ---------------------------
# Step 2: Train the model (cached so it only trains once)
# ---------------------------
@st.cache_resource
def train_model(data):
    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


model, accuracy = train_model(df)

st.info(f"This model is about **{accuracy * 100:.1f}%** accurate on test data.")

# ---------------------------
# Step 3: Let the user enter passenger details
# ---------------------------
st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2)
sex = st.selectbox("Sex", options=["male", "female"])
age = st.slider("Age", min_value=0, max_value=80, value=25)
fare = st.slider("Fare Paid ($)", min_value=0, max_value=300, value=30)

# ---------------------------
# Step 4: Make a prediction
# ---------------------------
if st.button("Predict Survival"):
    sex_number = 0 if sex == "male" else 1

    input_data = pd.DataFrame(
        [[pclass, sex_number, age, fare]],
        columns=["Pclass", "Sex", "Age", "Fare"],
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ This passenger would likely SURVIVE ({probability:.1f}% chance)")
    else:
        st.error(f"❌ This passenger would likely NOT survive ({probability:.1f}% chance of surviving)")
