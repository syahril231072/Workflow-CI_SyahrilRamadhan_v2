# ==========================================
# IMPORT LIBRARY
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "loan_approval_preprocessed.csv"
EXPERIMENT_NAME = (
    "Loan_Approval_Automation"
)

# ==========================================
# LOAD PREPROCESSED DATASET
# ==========================================

print(
    "Loading preprocessed dataset..."
)
import os

print("Current directory:", os.getcwd())
print("Files:", os.listdir())
df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset shape: {df.shape}"
)

# ==========================================
# FEATURE TARGET SPLIT
# ==========================================

X = df.drop(
    "loan_status",
    axis=1
)

y = df["loan_status"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# WAJIB SESUAI REVIEWER
mlflow.sklearn.autolog()

# ==========================================
# TRAINING
# ==========================================

with mlflow.start_run():

    print(
        "Training Random Forest model..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )
    # ======================================
    # EXPLICIT MODEL LOGGING
    # ======================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # ======================================
    # EVALUATION
    # ======================================

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\n========== RESULT ==========")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

print(
    "\nTraining completed successfully."
)

print(
    "Artifacts have been logged automatically by MLflow Autolog."
)