import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
heart_data = pd.read_csv('heart.csv')
X = heart_data.drop(columns='target')
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Get predictions on test set
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics for each model
print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS ON TEST SET")
print("=" * 80)
print()

models = {
    'Random Forest': (rf_pred, rf_proba),
    'Logistic Regression': (lr_pred, lr_proba),
    'XGBoost': (xgb_pred, xgb_proba)
}

metrics_data = {}

for name, (pred, proba) in models.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    
    metrics_data[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    }
    
    print(f"{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print()

# Calculate weights based on F1-Score (balanced metric)
print("=" * 80)
print("WEIGHT CALCULATION (Based on F1-Score)")
print("=" * 80)
print()

f1_scores = {name: metrics['F1-Score'] for name, metrics in metrics_data.items()}
total_f1 = sum(f1_scores.values())

weights = {name: score / total_f1 for name, score in f1_scores.items()}

print("F1-Scores:")
for name, f1 in f1_scores.items():
    print(f"  {name}: {f1:.4f}")
print()

print("Normalized Weights (sum = 1.0):")
for name, weight in weights.items():
    print(f"  {name}: {weight:.4f}")
print()

# Alternative: AUC-based weights
print("=" * 80)
print("ALTERNATIVE: WEIGHT CALCULATION (Based on AUC-ROC)")
print("=" * 80)
print()

auc_scores = {name: metrics['AUC-ROC'] for name, metrics in metrics_data.items()}
total_auc = sum(auc_scores.values())

auc_weights = {name: score / total_auc for name, score in auc_scores.items()}

print("AUC-ROC Scores:")
for name, auc in auc_scores.items():
    print(f"  {name}: {auc:.4f}")
print()

print("Normalized Weights (sum = 1.0):")
for name, weight in auc_weights.items():
    print(f"  {name}: {weight:.4f}")
print()

# Final recommendation
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("Use F1-Score based weights (better for imbalanced datasets):")
print()
print(f"WEIGHTS = {{")
print(f"    'RandomForest': {weights['Random Forest']:.4f},")
print(f"    'LogisticRegression': {weights['Logistic Regression']:.4f},")
print(f"    'XGBoost': {weights['XGBoost']:.4f}")
print(f"}}")
