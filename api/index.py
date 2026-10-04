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

# Model weights based on F1-Score analysis on test set
# Random Forest: 0.3384, Logistic Regression: 0.3298, XGBoost: 0.3318
MODEL_WEIGHTS = {
    "random_forest": 0.3384,
    "logistic_regression": 0.3298,
    "xgboost": 0.3318
}

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

        # Calculate weighted average prediction (based on F1-Score performance)
        weighted_disease_prob = (
            MODEL_WEIGHTS["random_forest"] * rf_disease_prob +
            MODEL_WEIGHTS["logistic_regression"] * lr_disease_prob +
            MODEL_WEIGHTS["xgboost"] * xgb_disease_prob
        )
        
        # Convert to 0-10 risk scale
        risk_score = (weighted_disease_prob / 100) * 10
        
        # Detect disagreement between models (if variance is high)
        disease_probs = [rf_disease_prob, lr_disease_prob, xgb_disease_prob]
        prob_variance = max(disease_probs) - min(disease_probs)
        has_disagreement = prob_variance > 25  # >25% difference = disagreement
        
        # Determine risk category
        if risk_score < 2.5:
            risk_category = "LOW RISK"
        elif risk_score < 5:
            risk_category = "MODERATE RISK"
        elif risk_score < 7.5:
            risk_category = "HIGH RISK"
        else:
            risk_category = "VERY HIGH RISK"
        
        result = "⚠️ Heart Disease Detected" if weighted_disease_prob > 50 else "✅ No Heart Disease"
        
        # Add weighted results to model_results
        model_results["weighted_ensemble"] = {
            "weighted_disease_prob": round(weighted_disease_prob, 2),
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
            "has_disagreement": has_disagreement,
            "prob_variance": round(prob_variance, 2)
        }

    return render_template(
        "index.html",
        result=result,
        values=values,
        model_results=model_results
    )

# For local testing with run_local.py
def handler(event, context):
    return app(event, context)
