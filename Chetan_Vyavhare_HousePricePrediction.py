"""
train_model.py
--------------
Trains a Gradient Boosting Regressor on the house price dataset,
evaluates it, and persists the model + feature scaler to disk.

Run:
    python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "data", "house_price_regression_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]
TARGET = "House_Price"

# ── Load & Split ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.describe().to_string())

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train ──────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train_sc, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
print(f"\nTest Metrics - MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R2: {r2:.4f}")

# ── Persist ────────────────────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(MODEL_DIR, "gbr_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
    json.dump(FEATURES, f)
print("Model, scaler, metrics and feature list saved to models/")

# ── Feature Importance Plot ────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="#3b82d4")
ax.set_title("Feature Importances (Gradient Boosting)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=120)
plt.close(fig)

# ── Actual vs Predicted Plot ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(y_test, y_pred, alpha=0.4, color="#3b82d4", edgecolors="none", s=25)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax2.set_xlabel("Actual Price ($)")
ax2.set_ylabel("Predicted Price ($)")
ax2.set_title("Actual vs Predicted House Prices")
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(MODEL_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close(fig2)

print("Plots saved to models/")
