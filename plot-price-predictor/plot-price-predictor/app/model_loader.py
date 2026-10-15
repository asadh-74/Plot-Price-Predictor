"""Loads all trained models once at startup and exposes a unified predict()."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

NUMERIC = ["acres", "distance_to_city_km", "frontage_m"]
CATEGORICAL = ["location", "zoning", "road_access", "corner_plot"]
COLUMNS = NUMERIC + CATEGORICAL
FEATURES = COLUMNS  # alias used by the API layer

MAPE = {"ridge": 0.5723, "random_forest": 0.1545,
        "gradient_boosting": 0.1319, "neural_net": 0.0991}  # from models/metrics.json


class MLP(nn.Module):
    """Must mirror the architecture used in training (see src/train.py)."""

    def __init__(self, input_dim: int, hidden=(128, 64, 32), dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(),
            nn.Linear(hidden[2], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ModelStore:
    def __init__(self):
        self.ridge = joblib.load(MODELS_DIR / "ridge_baseline.joblib")
        self.rf = joblib.load(MODELS_DIR / "rf_model.joblib")
        self.gbr = joblib.load(MODELS_DIR / "gbr_model.joblib")

        cfg = json.loads((MODELS_DIR / "nn_config.json").read_text())
        self.nn_pre = joblib.load(MODELS_DIR / "nn_preprocessor.joblib")
        self.nn_y_scaler = joblib.load(MODELS_DIR / "nn_y_scaler.joblib")
        self.nn = MLP(cfg["input_dim"], tuple(cfg["hidden"]), cfg["dropout"])
        self.nn.load_state_dict(torch.load(MODELS_DIR / "nn_model.pt", map_location="cpu"))
        self.nn.eval()

    def predict(self, features: dict) -> dict:
        row = pd.DataFrame([{c: features[c] for c in COLUMNS}])

        out = {
            "ridge": float(self.ridge.predict(row)[0]),
            "random_forest": float(self.rf.predict(row)[0]),
            "gradient_boosting": float(self.gbr.predict(row)[0]),
        }
        with torch.no_grad():
            x = torch.tensor(self.nn_pre.transform(row).astype(np.float32))
            log_p = self.nn_y_scaler.inverse_transform(
                self.nn(x).numpy().reshape(-1, 1)).squeeze(-1)
        out["neural_net"] = float(np.expm1(log_p)[0])
        return out
