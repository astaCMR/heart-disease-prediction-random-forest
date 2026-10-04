from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Load dataset (303 rows CSV)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "heart.csv")
heart_data = pd.read_csv(CSV_PATH)

X = heart_data.drop(columns="target")
y = heart_data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Train XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    values = {}
    model_results = {}

    if request.method == "POST":
        values = {col: request.form[col] for col in X.columns}
        data = [float(values[col]) for col in X.columns]

        # Get predictions from all three models with probability scores
        rf_proba = rf_model.predict_proba([data])[0]
        lr_proba = lr_model.predict_proba([data])[0]
        xgb_proba = xgb_model.predict_proba([data])[0]

        # Calculate disease probability (class 1)
        rf_disease_prob = rf_proba[1] * 100
        lr_disease_prob = lr_proba[1] * 100
        xgb_disease_prob = xgb_proba[1] * 100

        # Store results for each model
        model_results = {
            "random_forest": {
                "name": "Random Forest",
                "disease_prob": round(rf_disease_prob, 2),
                "healthy_prob": round(rf_proba[0] * 100, 2),
                "prediction": "⚠️ Disease Detected" if rf_proba[1] > 0.5 else "✅ No Disease"
            },
            "logistic_regression": {
                "name": "Logistic Regression",
                "disease_prob": round(lr_disease_prob, 2),
                "healthy_prob": round(lr_proba[0] * 100, 2),
                "prediction": "⚠️ Disease Detected" if lr_proba[1] > 0.5 else "✅ No Disease"
            },
            "xgboost": {
                "name": "XGBoost",
                "disease_prob": round(xgb_disease_prob, 2),
                "healthy_prob": round(xgb_proba[0] * 100, 2),
                "prediction": "⚠️ Disease Detected" if xgb_proba[1] > 0.5 else "✅ No Disease"
            }
        }

        # Calculate average prediction
        avg_disease_prob = (rf_disease_prob + lr_disease_prob + xgb_disease_prob) / 3
        result = "⚠️ Heart Disease Detected" if avg_disease_prob > 50 else "✅ No Heart Disease"

    return render_template(
        "index.html",
        result=result,
        values=values,
        model_results=model_results
    )

# For local testing with run_local.py
def handler(event, context):
    return app(event, context)
