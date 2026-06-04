# ==========================================
# IMPORT LIBRARY
# ==========================================

import os
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

# ==========================================
# LOAD PREPROCESSED DATASET
# ==========================================

print("Loading preprocessed dataset...")

print("Current directory:", os.getcwd())
print("Files:", os.listdir())

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# INPUT EXAMPLE FOR MODEL SIGNATURE
# ==========================================

input_example = X_train.iloc[:5]

# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri(
    f"file://{os.path.abspath('mlruns')}"
)

mlflow.sklearn.autolog()

# ==========================================
# TRAINING
# ==========================================

with mlflow.start_run():

    print("Training Random Forest model...")

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
        artifact_path="model",
        input_example=input_example
    )

    # ======================================
    # EVALUATION
    # ======================================

    y_pred = model.predict(X_test)

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

    # ======================================
    # LOG METRICS
    # ======================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
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

print("\nTraining completed successfully.")
print("Artifacts have been logged successfully.")