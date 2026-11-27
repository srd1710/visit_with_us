"""Tune, evaluate, track and serialize the customer purchase model."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_DIR / "artifacts" / "splits"
DEPLOY_DIR = PROJECT_DIR / "deployment"

TARGET = "ProdTaken"


# --------------------------------------------------
# Main training function
# --------------------------------------------------


def main():
    print("Starting Visit with Us model training...")

    # --------------------------------------------------
    # 1. Validate and load training/testing datasets
    # --------------------------------------------------

    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    for filename in required_files:
        file_path = SPLIT_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {file_path}")

    print(f"Loading datasets from: {SPLIT_DIR}")

    X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
    X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv")[TARGET]
    y_test = pd.read_csv(SPLIT_DIR / "y_test.csv")[TARGET]

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training target distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------
    # 2. Identify numeric and categorical features
    # --------------------------------------------------

    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X_train.columns if column not in numeric]

    print("Numeric features:", len(numeric))
    print("Categorical features:", len(categorical))

    # --------------------------------------------------
    # 3. Build preprocessing pipelines
    # --------------------------------------------------

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )

    # --------------------------------------------------
    # 4. Initialize XGBoost classifier
    # --------------------------------------------------

    scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    # --------------------------------------------------
    # 5. Hyperparameter tuning setup
    # --------------------------------------------------

    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.9],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --------------------------------------------------
    # 6. Configure MLflow
    # --------------------------------------------------

    # Use MLFLOW_TRACKING_URI if set; otherwise track to a local SQLite database
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visit-with-us-wellness-package")

    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # --------------------------------------------------
    # 7. Train and evaluate model
    # --------------------------------------------------

    with mlflow.start_run():
        print("Starting GridSearchCV hyperparameter tuning...")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "best_cv_f1": float(search.best_score_),
        }

        # --------------------------------------------------
        # 8. Log parameters and metrics to MLflow
        # --------------------------------------------------

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("target", TARGET)
        mlflow.set_tag("project", "Visit with Us")

        print("\nBest parameters:")
        print(json.dumps(search.best_params_, indent=2))

        print("\nTest metrics:")
        print(json.dumps(metrics, indent=2))

        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))

        # --------------------------------------------------
        # 9. Save trained model
        # --------------------------------------------------

        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

        model_path = DEPLOY_DIR / "model.joblib"
        joblib.dump(best_model, model_path)

        # --------------------------------------------------
        # 10. Generate Streamlit application metadata
        # --------------------------------------------------

        feature_info = {}

        for column in X_train.columns:
            series = X_train[column]
            has_values = series.notna().any()

            if pd.api.types.is_numeric_dtype(series):
                feature_info[column] = {
                    "type": "number",
                    "min": float(series.min()) if has_values else 0.0,
                    "max": float(series.max()) if has_values else 100.0,
                    "default": float(series.median()) if has_values else 0.0,
                }
            else:
                values = sorted(series.dropna().astype(str).unique().tolist())
                mode_values = series.mode(dropna=True)

                if not mode_values.empty:
                    default = str(mode_values.iloc[0])
                else:
                    default = values[0] if values else ""

                feature_info[column] = {
                    "type": "category",
                    "options": values,
                    "default": default,
                }

        metadata = {
            "target": TARGET,
            "features": feature_info,
            "metrics": metrics,
            "best_params": search.best_params_,
            "model_type": "XGBoost classifier",
        }

        metadata_path = DEPLOY_DIR / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # --------------------------------------------------
        # 11. Log model and metadata to MLflow
        # --------------------------------------------------

        mlflow.log_artifact(str(metadata_path))
        mlflow.sklearn.log_model(best_model, name="wellness_purchase_model")

        # --------------------------------------------------
        # 12. Final confirmation
        # --------------------------------------------------

        print("\nModel training completed successfully.")
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")

        print("\nGenerated deployment files:")
        for file in DEPLOY_DIR.iterdir():
            print(file.name)


if __name__ == "__main__":
    main()
