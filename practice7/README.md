# Breast Cancer Prediction API

ML system with FastAPI + Docker + MLflow + Streamlit + Batch Prediction Pipeline.

## Project Structure

```
sis3_breast/
├── train.py
├── main.py
├── streamlit_app.py
├── setup_db.py
├── batch_predict.py
├── model.joblib
├── scaler.joblib
├── feature_names.joblib
├── predictions.db
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

### Step 3 — Train model
```bash
python train.py
```

### Step 4 — Start FastAPI
```bash
uvicorn main:app --reload
```

### Step 5 — Start Streamlit
```bash
streamlit run streamlit_app.py
```

### Step 6 — Setup database and run batch pipeline
```bash
python setup_db.py
python batch_predict.py
```

---

## Docker

```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

---

## MLflow

| Type | Values |
|---|---|
| Parameters | n_estimators, max_depth, class_weight, random_state |
| Metrics | accuracy, f1_score, roc_auc |
| Artifacts | model.joblib, scaler.joblib, feature_names.joblib |
| Registry | breast-cancer-rf → @production |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/info` | Model info + metrics |
| POST | `/predict` | Get prediction |

---

## Batch Prediction Pipeline

| Component | Details |
|---|---|
| Database | SQLite — predictions.db |
| Tables | input_data, predictions |
| Schedule | Every 5 minutes |
| Scheduler | Python schedule library |

---

## Results

- **Accuracy:** 94.74%
- **ROC-AUC:** 99.40%
- **F1 Score:** 95.83%