
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

MODEL_PATH = "model.pkl"
DATA_PATH  = "heart_failure_clinical_records_dataset (1).csv"

def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop("DEATH_EVENT", axis=1), df["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    print(f"Trained model with accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
    pickle.dump(model, open(MODEL_PATH, "wb"))
    return model, X.columns.tolist()


if os.path.exists(MODEL_PATH):
    model           = pickle.load(open(MODEL_PATH, "rb"))
    feature_order   = pd.read_csv(DATA_PATH, nrows=0).drop("DEATH_EVENT", axis=1).columns.tolist()
else:
    model, feature_order = train_and_save_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = np.array([[float(request.form[col]) for col in feature_order]])
        prediction = model.predict(features)[0]
        result = "High Risk of Death" if prediction else "Low Risk of Death"
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)