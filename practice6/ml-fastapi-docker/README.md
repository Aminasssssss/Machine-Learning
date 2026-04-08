# Breast Cancer Prediction API

ML API for breast cancer classification using Random Forest and FastAPI, deployed with Docker.

## Dataset
- **Source:** Breast Cancer Wisconsin Dataset (sklearn)
- **Task:** Binary classification — malignant (0) or benign (1)
- **Features:** 30 numerical features (cell measurements)
- **Model:** Random Forest Classifier (class_weight='balanced')

## Project Structure
ml-fastapi-docker/
├── train.py
├── main.py
├── model.joblib
├── scaler.joblib
├── feature_names.joblib
├── requirements.txt
├── Dockerfile
└── README.md

## How to Run

### 1. Train the model
```bash
python train.py
```

### 2. Run locally
```bash
uvicorn main:app --reload
```

### 3. Run with Docker
```bash
docker build -t ml-fastapi-app .
docker run -p 8000:8000 ml-fastapi-app
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Get prediction |

## Test the API
Open `http://localhost:8000/docs` for Swagger UI.

### Example request:
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 
               0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 
               8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 
               0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 
               0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

### Example response:
```json
{
  "prediction": 0,
  "diagnosis": "malignant",
  "probability": {
    "malignant": 0.97,
    "benign": 0.03
  }
}
```

## Results
- **Accuracy:** 94.74%
- **ROC-AUC:** 99.40%
- **F1 Score:** 95.83%