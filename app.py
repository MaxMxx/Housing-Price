from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Chargement du modèle au démarrage de l'API
model = joblib.load("model.joblib")

app = FastAPI(
    title="Housing Price Prediction API",
    description="API California Housing",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Revenu médian")
    HouseAge: float = Field(..., description="Âge moyen des logements")
    AveRooms: float = Field(..., description="Nombre moyen de pièces")
    AveBedrms: float = Field(..., description="Nombre moyen de chambres")
    Population: float = Field(..., description="Population du quartier")
    AveOccup: float = Field(..., description="Occupation moyenne")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3,
                "HouseAge": 41,
                "AveRooms": 6.9,
                "AveBedrms": 1.0,
                "Population": 322,
                "AveOccup": 2.5,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }


def _features_to_row(features: HouseFeatures) -> list:
    return [
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]


@app.get("/health")
def health():
    """Vérifie que l'API fonctionne."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    """Prédit la valeur médiane d'un logement à partir de ses caractéristiques."""
    row = _features_to_row(features)
    prediction = model.predict([row])[0]
    return {"predicted_house_value": round(float(prediction), 2)}


@app.post("/predict_batch")
def predict_batch(features_list: List[HouseFeatures]):
    """Prédit la valeur médiane pour plusieurs logements en une seule requête (bonus)."""
    rows = [_features_to_row(f) for f in features_list]
    predictions = model.predict(rows)
    return {
        "predictions": [
            {"predicted_house_value": round(float(p), 2)} for p in predictions
        ]
    }
