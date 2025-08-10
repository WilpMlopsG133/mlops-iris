# mlops-iris/app/main.py

import os
import logging
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Part 3: API & Docker Packaging ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Initialize FastAPI application
app = FastAPI(
    title="Iris MLOps API",
    description="A minimal MLOps pipeline for the Iris dataset.",
    version="1.0.0",
)

# Use the environment variable to set the tracking URI inside the container
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load the best model from the MLflow Model Registry
try:
    logged_model = mlflow.sklearn.load_model("models:/IrisBestModel/Production")
    logger.info("Successfully loaded model from MLflow Model Registry.")
except Exception as e:
    logger.error(f"Failed to load model from MLflow Model Registry: {e}. Please ensure the model has been trained and registered.")
    logged_model = None

@app.get("/")
def home():
    return {"message": "Welcome to the Iris MLOps API. Use the /predict endpoint for predictions."}

@app.post("/predict")
def predict(features: IrisFeatures):
    """Endpoint for making predictions on Iris data."""
    if logged_model is None:
        return {"error": "Model not available. Please check server logs."}

    logger.info(f"Received prediction request with data: {features.dict()}")
    input_df = pd.DataFrame([features.dict()])
    prediction = logged_model.predict(input_df)[0]

    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_map.get(prediction, 'unknown')

    logger.info(f"Prediction result: {predicted_species}")

    return {"prediction": predicted_species}

# Expose a basic /metrics endpoint (Optional)
@app.get("/metrics")
def get_metrics():
    return {"message": "Metrics endpoint (not fully implemented for this demo)"}

