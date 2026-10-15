"""FastAPI service: plot price estimation with model comparison.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model_loader import MAPE, ModelStore
from app.schemas import ModelName, PlotFeatures, PredictionResponse

store: ModelStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = ModelStore()          # load artifacts once, not per request
    yield


app = FastAPI(
    title="Plot Price Predictor",
    description="ML/DL API that estimates land plot prices from size (acres), "
                "location, zoning, road access and frontage. Serves four models "
                "(Ridge baseline, Random Forest, Gradient Boosting, PyTorch MLP) "
                "and returns the ensemble comparison.",
    version="1.0.0",
    lifespan=lifespan,
)

RECOMMENDED: ModelName = "neural_net"   # best test MAPE (see models/metrics.json)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": store is not None}


@app.get("/models")
def models():
    return {
        "available": list(MAPE.keys()),
        "test_mape": MAPE,
        "recommended": RECOMMENDED,
        "features": {
            "numeric": ["acres", "distance_to_city_km", "frontage_m"],
            "categorical": ["location", "zoning", "road_access", "corner_plot"],
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PlotFeatures):
    preds = store.predict(features.model_dump())
    best = preds[RECOMMENDED]
    mape = MAPE[RECOMMENDED]
    return PredictionResponse(
        input=features,
        predictions={k: round(v, 0) for k, v in preds.items()},
        recommended_model=RECOMMENDED,
        estimated_price=round(best, 0),
        price_range={
            "low": round(best * (1 - mape), 0),
            "high": round(best * (1 + mape), 0),
        },
    )
