"""Loads all trained models once at startup and exposes a unified predict()."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

NUMERIC = ["acres", "distance_to_city_km", "frontage_m"]
CATEGORICAL = ["location", "zoning", "road_access", "corner_plot"]
COLUMNS = NUMERIC + CATEGORICAL
FEATURES = COLUMNS  # alias used by the API layer

MAPE = {"ridge": 0.5723, "random_forest": 0.1545,
        "gradient_boosting": 0.1319, "neural_net": 0.0991}  # from models/metrics.json




class ModelStore:
    def __init__(self):
        self.ridge = joblib.load(MODELS_DIR / "ridge_baseline.joblib")
        self.rf = joblib.load(MODELS_DIR / "rf_model.joblib")
        self.gbr = joblib.load(MODELS_DIR / "gbr_model.joblib")

        cfg = json.loads((MODELS_DIR / "nn_config.json").read_text())
        self.nn_pre = joblib.load(MODELS_DIR / "nn_preprocessor.joblib")
        self.nn_y_scaler = joblib.load(MODELS_DIR / "nn_y_scaler.joblib")
        self.nn_weights = {k: np.array(v, dtype=np.float32) for k,v in
                           json.loads((MODELS_DIR / "nn_weights.json").read_text()).items()}

    def predict(self, features: dict) -> dict:
        row = pd.DataFrame([{c: features[c] for c in COLUMNS}])

        out = {
            "ridge": float(self.ridge.predict(row)[0]),
            "random_forest": float(self.rf.predict(row)[0]),
            "gradient_boosting": float(self.gbr.predict(row)[0]),
        }
        x = self.nn_pre.transform(row).astype(np.float32)
        for layer in [0, 3, 6, 8]:
            x = x @ self.nn_weights[f"net.{layer}.weight"].T + self.nn_weights[f"net.{layer}.bias"]
            if layer != 8:
                x = np.maximum(x, 0)
        log_p = self.nn_y_scaler.inverse_transform(x.reshape(-1, 1)).squeeze(-1)
        out["neural_net"] = float(np.expm1(log_p)[0])
        return out
