# Machine Learning Project #2 — Titanic Survival Predictor

A simple, beginner-friendly classification model that predicts whether
a Titanic passenger would have survived, based on their class, sex,
age, and fare.

## How this is different from Project #1
- Project #1 (house price prediction): **regression** — predicts a
  number (a price).
- Project #2 (this one): **classification** — predicts a category
  (survived / did not survive).

## What it does
- Trains a Logistic Regression model on your Titanic-dataset.csv
- Shows the model's accuracy on unseen test data
- Lets you enter a passenger's class, sex, age, and fare
- Predicts survival with a confidence percentage

## How to run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
**Important:** `Titanic-dataset.csv` must be in the same folder as
`app.py` on GitHub, or the app won't find it.

1. Push `app.py`, `requirements.txt`, and `Titanic-dataset.csv` to a
   GitHub repo.
2. Go to https://share.streamlit.io and connect your repo.
3. Set `app.py` as the main file.
4. Deploy.

## How it works
- Only 4 simple features are used: Pclass, Sex, Age, Fare — kept
  simple on purpose.
- Missing ages are filled with the average age.
- `Sex` is converted to numbers (male=0, female=1) since ML models
  need numeric input.
- `LogisticRegression` is a simple, beginner-friendly classification
  algorithm — good first choice before trying more complex models.
- `@st.cache_resource` means the model only trains once, not every
  time you click "Predict."
