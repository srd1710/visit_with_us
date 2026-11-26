"""Tune, evaluate, track and serialize the customer purchase model."""
from pathlib import Path
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SPLIT_DIR = REPO_ROOT / "artifacts" / "splits"
DEPLOY_DIR = ROOT / "deployment"
TARGET = "ProdTaken"

def main():
    X_train = pd.read_csv(SPLIT_DIR / "Xtrain.csv")
    X_test = pd.read_csv(SPLIT_DIR / "Xtest.csv")
    y_train = pd.read_csv(SPLIT_DIR / "ytrain.csv")[TARGET]
    y_test = pd.read_csv(SPLIT_DIR / "ytest.csv")[TARGET]

    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.9],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visit-with-us-wellness-package")
    with mlflow.start_run():
        search = GridSearchCV(
            pipeline, param_grid, scoring="f1", cv=cv, n_jobs=1,
            refit=True, verbose=1
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
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("target", TARGET)
        print("Best parameters:", search.best_params_)
        print("Test metrics:", json.dumps(metrics, indent=2))
        print(classification_report(y_test, predictions, zero_division=0))

        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, DEPLOY_DIR / "model.joblib")
        defaults = {}
        feature_info = {}
        for col in X_train.columns:
            series = X_train[col]
            if pd.api.types.is_numeric_dtype(series):
                median = float(series.median()) if series.notna().any() else 0.0
                defaults[col] = median
                feature_info[col] = {
                    "type": "number",
                    "min": float(series.min()) if series.notna().any() else 0.0,
                    "max": float(series.max()) if series.notna().any() else 100.0,
                    "default": median,
                }
            else:
                values = sorted(series.dropna().astype(str).unique().tolist())
                default = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else (values[0] if values else "")
                defaults[col] = default
                feature_info[col] = {
                    "type": "category", "options": values, "default": default
                }
        metadata = {
            "target": TARGET,
            "features": feature_info,
            "metrics": metrics,
            "best_params": search.best_params_,
            "model_type": "XGBoost classifier",
        }
        (DEPLOY_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(DEPLOY_DIR / "model_metadata.json"))
        mlflow.sklearn.log_model(best_model, name="wellness_purchase_model")
        print(f"Model saved to {DEPLOY_DIR / 'model.joblib'}")

if __name__ == "__main__":
    main()
