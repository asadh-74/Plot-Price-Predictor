"""Train and compare 4 models for plot price prediction.

Usage:  python src/train.py

Outputs (saved to models/):
    ridge_baseline.joblib, rf_model.joblib, gbr_model.joblib
    nn_preprocessor.joblib, nn_y_scaler.joblib, nn_model.pt, nn_config.json
    metrics.json

Notes
-----
* XGBoost can replace HistGradientBoostingRegressor by uncommenting the import
  below and swapping the estimator — the preprocessing pipeline is identical.
* The NN predicts log1p(price) with a standardized target, which stabilises
  training on the right-skewed price distribution.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from xgboost import XGBRegressor  # optional, drop-in replacement for HGB

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "plot_listings.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

NUMERIC = ["acres", "distance_to_city_km", "frontage_m"]
CATEGORICAL = ["location", "zoning", "road_access", "corner_plot"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64, 32), dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(),
            nn.Linear(hidden[2], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 0),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 0),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def main():
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL]
    y = df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    results = {}

    sk_models = {
        "ridge": (Ridge(alpha=1.0), "ridge_baseline.joblib"),
        "random_forest": (RandomForestRegressor(400, random_state=42, n_jobs=-1), "rf_model.joblib"),
        "gradient_boosting": (HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.06, random_state=42), "gbr_model.joblib"),
        # "gradient_boosting": (XGBRegressor(n_estimators=600, learning_rate=0.06,
        #                                    max_depth=6, subsample=0.9, random_state=42), "gbr_model.joblib"),
    }
    for name, (est, fname) in sk_models.items():
        pipe = Pipeline([("pre", pre), ("m", est)]).fit(X_tr, y_tr)
        results[name] = metrics(y_te, pipe.predict(X_te))
        joblib.dump(pipe, MODELS / fname, compress=3)
        print(f"{name:20s} {results[name]}")

    # ---- Neural net ----
    Xtr = pre.fit_transform(X_tr).astype(np.float32)
    Xte = pre.transform(X_te).astype(np.float32)
    ytr_log = np.log1p(y_tr.values.astype(np.float32))
    yte_log = np.log1p(y_te.values.astype(np.float32))
    y_scaler = StandardScaler().fit(ytr_log.reshape(-1, 1))
    ytr_s = y_scaler.transform(ytr_log.reshape(-1, 1)).astype(np.float32).squeeze(-1)

    torch.manual_seed(42)
    net = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr_s)
    yte_t = torch.tensor(y_scaler.transform(yte_log.reshape(-1, 1)).astype(np.float32).squeeze(-1))

    best, wait, patience = float("inf"), 0, 40
    for epoch in range(400):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xtr_t), ytr_t)
        loss.backward(); opt.step()
        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                val = lossf(net(torch.tensor(Xte)), yte_t)
            if val < best:
                best, wait = val, 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"early stop @ epoch {epoch}")
                    break

    net.eval()
    with torch.no_grad():
        p_log = y_scaler.inverse_transform(
            net(torch.tensor(Xte)).numpy().reshape(-1, 1)).squeeze(-1)
    results["neural_net"] = metrics(y_te, np.expm1(p_log))
    print(f"{'neural_net':20s} {results['neural_net']}")

    joblib.dump(pre, MODELS / "nn_preprocessor.joblib")
    joblib.dump(y_scaler, MODELS / "nn_y_scaler.joblib")
    torch.save(net.state_dict(), MODELS / "nn_model.pt")
    (MODELS / "nn_config.json").write_text(json.dumps(
        {"input_dim": int(Xtr.shape[1]), "hidden": [128, 64, 32], "dropout": 0.15}))
    (MODELS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nArtifacts + metrics.json saved to {MODELS}")


if __name__ == "__main__":
    main()
