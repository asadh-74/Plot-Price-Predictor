"""Request/response schemas for the price estimation API."""
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field

Location = Literal["Downtown", "Riverside", "Lakeview", "Old Town",
                   "Meadowbrook", "Hillcrest", "Greenfield", "Industrial Zone"]
Zoning = Literal["residential", "commercial", "agricultural"]
RoadAccess = Literal["paved", "gravel", "dirt"]
ModelName = Literal["ridge", "random_forest", "gradient_boosting", "neural_net"]


class PlotFeatures(BaseModel):
    acres: float = Field(..., gt=0, le=10_000, description="Plot size in acres")
    location: Location
    zoning: Zoning = "residential"
    road_access: RoadAccess = "paved"
    corner_plot: bool = False
    distance_to_city_km: float = Field(10.0, ge=0, le=500)
    frontage_m: float = Field(20.0, gt=0, le=500)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "acres": 2.5, "location": "Lakeview", "zoning": "residential",
                "road_access": "paved", "corner_plot": True,
                "distance_to_city_km": 4.0, "frontage_m": 30.0
            }]
        }
    }


class PredictionResponse(BaseModel):
    input: PlotFeatures
    predictions: Dict[ModelName, float]
    recommended_model: ModelName
    estimated_price: float
    price_range: Dict[str, float]  # ± MAPE of the recommended model
    currency: str = "USD"
