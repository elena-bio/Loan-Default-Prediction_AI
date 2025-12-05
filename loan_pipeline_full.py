# loan_pipeline_full.py
"""
Robust full pipeline for the Loan Default project (Enhanced Version).

Updates by Elham's Team:
- Removed feature selection limit (uses all valid categorical features).
- Added XGBoost Classifier for better performance.
- Enhanced Model Evaluation logic.

Run:
    python loan_pipeline_full.py
"""
import os
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Modeling
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc, classification_report)
import joblib

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed. Skipping XGBoost model. (pip install xgboost)")

# -------------------------
# Settings: expected names and candidate search dirs
# -------------------------
EXPECTED_TRAIN = "train_LZV4RXX.csv"
TEST_PREFIX = "test"
TEST_EXT = ".csv"

CANDIDATE_DIRS = [
    Path.cwd(),
    Path.cwd() / "data",
    Path("/home/elena/Documents/AI Redi School"),
    Path("/mnt/data"),
]

def find_file_exact(filename):
    for base in CANDIDATE_DIRS:
        try:
            if not base.exists(): continue
        except Exception: continue
        candidate = base / filename
        if candidate.exists():
            return candidate.resolve()
    for p in Path.cwd().iterdir():
        if p.name == filename:
            return p.resolve()
    return None

def find_test_by_prefix(prefix=TEST_PREFIX, ext=TEST_EXT):
    for base in CANDIDATE_DIRS:
        try:
            if not base.exists(): continue
        except Exception: continue
        for p in base.iterdir():
            if p.is_file() and p.name.lower().startswith(prefix.lower()) and p.name.lower().endswith(ext.lower()):
                return p.resolve()
    for p in Path.cwd().iterdir():
        if p.is_file() and p.name.lower().startswith(prefix.lower()) and p.name.lower().endswith(ext.lower()):
            return p.resolve()
    return None

def find_or_exit_train():
    p = find_file_exact(EXPECTED_TRAIN)
    if p is None:
        print(f"\nERROR: Could not find TRAIN file: {EXPECTED_TRAIN}")
        sys.exit(1)
    print("Found train at:", p)
    return str(p)

def find_or_exit_test():
    p = find_test_by_prefix()
    if p is None:
        print("\nERROR: Could not find any test CSV.")
        sys.exit(1)
    print("Auto-detected test file at:", p)
    return str(p)

# -------------------------
# 1. Locate files & Load
# -------------------------
train_path = find_or_exit_train()
test_path = find_or_exit_test()

print("\nLoading CSV files...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# -------------------------
# 2. Initial Checks & Cleaning
# -------------------------
TARGET = "loan_default"
if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

train_orig = train.copy()
test_orig = test.copy()

print("\n=== Cleaning ===")
train.columns = [c.strip() for c in train.columns]
test.columns = [c.strip() for c in test.columns]

# Drop duplicates
before = train.shape[0]
train = train.drop_duplicates()
print(f"Dropped {before - train.shape[0]} duplicate rows.")

# Numeric conversions
if "age" in train.columns:
    train["age"] = pd.to_numeric(train["age"], errors="coerce").round().astype("Int64")
    if "age" in test.columns:
        test["age"] = pd.to_numeric(test.get("age"), errors="coerce").round().astype("Int64")

# Strip categorical strings
for c in train.select_dtypes(include=['object']).columns:
    train[c] = train[c].astype(str).str.strip()
    if c in test.columns:
        test[c] = test[c].astype(str).str.strip()

# Outlier flags (IQR)
num_cols_raw = [c for c in train.select_dtypes(include=[np.number]).columns.tolist() if c != TARGET]
for c in num_cols_raw:
    q1 = train[c].quantile(0.25)
    q3 = train[c].quantile(0.75)
    iqr = q3 - q1
    flag_col = f"{c}_outlier_flag"
    if pd.isna(iqr) or iqr == 0:
        continue
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    train[flag_col] = ((train[c] < low) | (train[c] > high)).astype(int)
    if c in test.columns:
        test[flag_col] = ((test[c] < low) | (test[c] > high)).astype(int)

train.to_csv("cleaned_loan_data_final.csv", index=False)
print("Saved cleaned file.")

# -------------------------
# 3. Feature Engineering
# -------------------------
print("\n=== Feature engineering ===")
# Example: Loan to Asset Ratio
if "loan_amount" in train.columns and "asset_cost" in train.columns:
    train["loan_to_asset"] = train["loan_amount"] / train["asset_cost"].replace(0, np.nan)
    test["loan_to_asset"]  = test.get("loan_amount", 0) / test.get("asset_cost", pd.Series(np.nan)).replace(0, np.nan)
    
    # Log transforms
    train["loan_amount_log"] = np.log1p(train["loan_amount"].fillna(0))
    test["loan_amount_log"]  = np.log1p(test.get("loan_amount", 0).fillna(0))

print("Feature engineering completed.")

# -------------------------
# 4. EDA (Simplified for brevity)
# -------------------------
print("\n=== EDA ===")
os.makedirs("plots", exist_ok=True)
sns.set(style="whitegrid")

# Just one example plot to ensure pipeline works
if "loan_amount" in train.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(train["loan_amount"].dropna(), bins=30)
    plt.title("Distribution of Loan Amount")
    plt.savefig("plots/hist_loan_amount.png")
    plt.close()

with open("Elham_EDA_Report.md", "w", encoding="utf-8") as f:
    f.write(f"# Loan Default Report\n\nTrain shape: {train.shape}\n")
    f.write("Plots saved in plots/ folder.\n")

# -------------------------
# 5. Prepare Model-Ready Data
# -------------------------
print("\n=== Prepare model-ready dataset ===")
# FIX: Removing the limit on categorical features!
numeric_features = [c for c in train.select_dtypes(include=[np.number]).columns.tolist() if c != TARGET]
categorical_candidates = train.select_dtypes(include=['object']).columns.tolist()

# Apply a reasonable limit only if distinct values are too high (e.g. > 50 categories), 
# but DO NOT limit the NUMBER of columns arbitrarily.
categorical_features = [c for c in categorical_candidates if train[c].nunique() <= 50]

print(f"Selected {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

selected_features = numeric_features + categorical_features
model_ready = train[selected_features + [TARGET]].copy()
model_ready.to_csv("model_ready_loan_data.csv", index=False)

# -------------------------
# 6. Modeling (Pipeline)
# -------------------------
print("\n=== Modeling ===")
X = model_ready.drop(columns=[TARGET])
y = model_ready[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing Pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preproc = ColumnTransformer([
    ("num", num_pipe, numeric_features), 
    ("cat", cat_pipe, categorical_features)
], remainder="drop")

# Define Models
models = {
    "LogisticRegression": Pipeline([
        ("preproc", preproc), 
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "RandomForest": Pipeline([
        ("preproc", preproc), 
        ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
}

if XGB_AVAILABLE:
    models["XGBoost"] = Pipeline([
        ("preproc", preproc),
        ("clf", XGBClassifier(n_estimators=200, learning_rate=0.05, n_jobs=-1, eval_metric='logloss', random_state=42))
    ])

# Training and Evaluation
eval_results = {}
best_score = -1
best_model_name = None
best_pipe = None

print(f"Training {len(models)} models...")

for name, pipe in models.items():
    print(f" -> Training {name}...")
    pipe.fit(X_tr, y_tr)
    
    # Predict
    preds = pipe.predict(X_val)
    try:
        probs = pipe.predict_proba(X_val)[:,1]
    except:
        probs = None
        
    # Metrics
    acc = accuracy_score(y_val, preds)
    roc = roc_auc_score(y_val, probs) if probs is not None else 0.5
    f1 = f1_score(y_val, preds, zero_division=0)
    
    eval_results[name] = {"accuracy": acc, "roc_auc": roc, "f1": f1}
    print(f"    {name}: AUC={roc:.3f}, F1={f1:.3f}")

    # Logic to select best model (prioritizing ROC-AUC)
    if roc > best_score:
        best_score = roc
        best_model_name = name
        best_pipe = pipe

print(f"\nBest Model Selected: {best_model_name} (AUC={best_score:.3f})")

# Save detailed metrics
with open("plots/eval_metrics_summary.json", "w") as fh:
    json.dump(eval_results, fh, indent=2)

# -------------------------
# 7. Retrain & Submission
# -------------------------
print(f"\nRetraining best model ({best_model_name}) on FULL dataset...")
best_pipe.fit(X, y)
joblib.dump(best_pipe, "best_loan_model.joblib")

print("Generating submission...")
# Prepare Test Data
for f in selected_features:
    if f not in test.columns:
        test[f] = np.nan

X_test_ready = test[selected_features]
test_preds = best_pipe.predict(X_test_ready)
test_probs = best_pipe.predict_proba(X_test_ready)[:,1] if hasattr(best_pipe, "predict_proba") else np.zeros(len(test_preds))

submission = pd.DataFrame({
    "loan_id": test_orig["loan_id"] if "loan_id" in test_orig.columns else range(len(test_preds)),
    "loan_default_pred": test_preds,
    "prob": test_probs
})
submission.to_csv("submission.csv", index=False)

print("\nPipeline Complete! Outputs:")
print("- best_loan_model.joblib")
print("- submission.csv")
print("- plots/ folder")