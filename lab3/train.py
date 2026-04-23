import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_NAME = "breast-cancer-rf"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

params = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": 42,
    "class_weight": "balanced",
}

with mlflow.start_run(run_name="kfold-validation-run"):
    mlflow.log_params(params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
   
    cv_results = cross_validate(
        RandomForestClassifier(**params), 
        X_train_scaled, 
        y_train, 
        cv=skf, 
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=False
    )

    mlflow.log_metric("cv_accuracy_mean", round(cv_results['test_accuracy'].mean(), 4))
    mlflow.log_metric("cv_f1_mean", round(cv_results['test_f1'].mean(), 4))
    mlflow.log_metric("cv_roc_auc_mean", round(cv_results['test_roc_auc'].mean(), 4))
    mlflow.log_metric("cv_accuracy_std", round(cv_results['test_accuracy'].std(), 4))

    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    final_metrics = {
        "final_accuracy":  accuracy_score(y_test, y_pred),
        "final_f1":        f1_score(y_test, y_pred),
        "final_roc_auc":   roc_auc_score(y_test, y_prob)
    }
    mlflow.log_metrics({k: round(v, 4) for k, v in final_metrics.items()})

    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_names, "feature_names.joblib")

    mlflow.log_artifact("model.joblib")
    mlflow.log_artifact("scaler.joblib")
    mlflow.log_artifact("feature_names.joblib")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    print(f"K-Fold F1 Mean: {cv_results['test_f1'].mean():.4f}")
    print(f"Final Test F1: {final_metrics['final_f1']:.4f}")

try:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest_version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(MODEL_NAME, "production", str(latest_version))
        print(f"Version {latest_version} is now @production")
except Exception as e:
    print(f"Registry error: {e}")
    