# Breast Cancer Prediction API — SIS-3

ML API with FastAPI + Docker + MLflow + Streamlit.

## Project Structure
```
ml-fastapi-docker/
├── train.py              ← Train model with K-Fold + MLflow logging
├── main.py               ← FastAPI endpoints
├── streamlit_app.py      ← Streamlit frontend (NEW — SIS-3)
├── model.joblib
├── scaler.joblib
├── feature_names.joblib
├── requirements.txt
├── Dockerfile
└── README.md
```

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start MLflow server
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000 to see MLflow UI.

### Step 3 — Train model (logs to MLflow automatically)
```bash
python train.py
```

### Step 4 — Start FastAPI
```bash
uvicorn main:app --reload
```
Open http://localhost:8000/docs for Swagger UI.

### Step 5 — Start Streamlit frontend
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501

---

## Docker (FastAPI only)
```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

---

## MLflow — What gets logged

| Type | Values |
|---|---|
| Parameters | n_estimators, max_depth, class_weight, random_state |
| Metrics | accuracy, precision, recall, f1_score, roc_auc |
| Artifacts | model.joblib, scaler.joblib, feature_names.joblib |
| Registry | breast-cancer-rf → version 8 → @production |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/info` | Model info + metrics |
| POST | `/predict` | Get prediction |

## Results
- **Accuracy:** 94.74%
- **ROC-AUC:** 99.40%
- **F1 Score:** 95.83%
