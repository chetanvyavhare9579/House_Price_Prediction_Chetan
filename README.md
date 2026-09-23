# 🏠 House Price Prediction

A full-stack machine learning web app that predicts house prices using a **Gradient Boosting Regressor**, with a **Flask REST API** backend and a **Streamlit** frontend.

---

## 📁 Project Structure

```
house_price_prediction/
├── data/
│   └── house_price_regression_dataset.csv   # Dataset[https://www.kaggle.com/datasets/prokshitha/home-value-insights?resource=download] (1000 rows, 8 columns)
├── models/                                   # Auto-generated after training
│   ├── gbr_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   ├── features.json
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
├── train_model.py      # ML training script
├── app.py              # Flask REST API backend
├── streamlit_app.py    # Streamlit frontend
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Train a Gradient Boosting Regressor
- Print test metrics (MAE, RMSE, R²)
- Save the model, scaler, metrics, and plots to `models/`

### 3. Start the Flask backend

```bash
python app.py
```

API runs at `http://localhost:5000`

### 4. Launch the Streamlit frontend (in a new terminal)

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

> **Note:** The Streamlit app also works offline (loads model artifacts locally) if the Flask API is not running.

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                     |
|--------|-----------------------|---------------------------------|
| GET    | `/health`             | Service liveness check          |
| POST   | `/predict`            | Predict house price (JSON body) |
| GET    | `/metrics`            | Model evaluation metrics        |
| GET    | `/feature_importance` | Feature importance scores       |

### Sample `/predict` request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Square_Footage": 2500,
    "Num_Bedrooms": 3,
    "Num_Bathrooms": 2,
    "Year_Built": 2010,
    "Lot_Size": 3.0,
    "Garage_Size": 1,
    "Neighborhood_Quality": 7
  }'
```

---

## 🧠 Model

| Parameter        | Value                       |
|------------------|-----------------------------|
| Algorithm        | Gradient Boosting Regressor |
| n_estimators     | 300                         |
| learning_rate    | 0.05                        |
| max_depth        | 5                           |
| subsample        | 0.8                         |
| Feature Scaling  | StandardScaler              |
| Train/Test Split | 80% / 20%                   |

---

## 📊 Features Used

| Feature               | Description                         |
|-----------------------|-------------------------------------|
| Square_Footage        | Total living area (sq ft)           |
| Num_Bedrooms          | Number of bedrooms                  |
| Num_Bathrooms         | Number of bathrooms                 |
| Year_Built            | Year the house was constructed      |
| Lot_Size              | Lot size in acres                   |
| Garage_Size           | Garage capacity (number of cars)    |
| Neighborhood_Quality  | Quality score (1–10)                |

---

## 🖥️ Frontend Tabs

| Tab              | Content                                                        |
|------------------|----------------------------------------------------------------|
| 🔮 Predict       | Interactive sliders + prediction result card                   |
| 📊 Data Analysis | EDA: histogram, scatter, bar, heatmap, box plots               |
| 🤖 Model         | Metrics, feature importance, actual vs predicted scatter plot  |
