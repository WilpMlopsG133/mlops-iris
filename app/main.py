import logging
import os
import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Part 1: Dataset Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads the Iris dataset and preprocesses it."""
    logger.info("Loading and preprocessing data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# --- Part 2: Model Training & Experiment Tracking ---
def train_and_track_models(X_train, y_train, X_test, y_test):
    """Trains multiple models and tracks experiments with MLflow."""
    logger.info("Starting MLflow experiment...")
    
    # Set the MLflow tracking URI to a local directory for this assignment
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("Iris_Model_Training")

    best_accuracy = 0
    best_model_uri = ""
    
    # Train Logistic Regression model
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(solver='liblinear', random_state=42)
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_param("solver", 'liblinear')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(lr, "logistic_regression_model")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_uri = mlflow.get_artifact_uri("logistic_regression_model")
        
        logger.info(f"Logistic Regression Accuracy: {accuracy}")

    # Train RandomForest Classifier model
    with mlflow.start_run(run_name="Random_Forest"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(rf, "random_forest_model")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_uri = mlflow.get_artifact_uri("random_forest_model")
            
        logger.info(f"Random Forest Accuracy: {accuracy}")

    # Register the best model to the MLflow Model Registry
    logger.info("Registering the best model...")
    mlflow.register_model(best_model_uri, "IrisBestModel")
    logger.info(f"Best model registered from: {best_model_uri}")
    
    return best_model_uri

# --- Part 3: API & Docker Packaging ---
# Define a Pydantic model for input validation (Bonus)
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

# Load the best model from the MLflow Model Registry
try:
    # Use "production" stage to get the latest registered model
    logged_model = mlflow.sklearn.load_model("models:/IrisBestModel/Production")
    logger.info("Successfully loaded model from MLflow Model Registry.")
except Exception as e:
    logger.error(f"Failed to load model from MLflow Model Registry: {e}. Model will be re-trained.")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    best_model_uri = train_and_track_models(X_train, y_train, X_test, y_test)
    logged_model = mlflow.sklearn.load_model(best_model_uri)
    logger.info("New model trained and loaded.")

@app.get("/")
def home():
    return {"message": "Welcome to the Iris MLOps API. Use the /predict endpoint for predictions."}

@app.post("/predict")
def predict(features: IrisFeatures):
    """Endpoint for making predictions on Iris data."""
    # --- Part 5: Logging ---
    logger.info(f"Received prediction request with data: {features.dict()}")
    
    # Convert input to a format the model can use
    input_df = pd.DataFrame([features.dict()])
    
    # Make a prediction
    prediction = logged_model.predict(input_df)[0]
    
    # Map the numerical prediction back to a species name
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = species_map.get(prediction, 'unknown')
    
    # --- Part 5: Logging ---
    logger.info(f"Prediction result: {predicted_species}")
    
    return {"prediction": predicted_species}

# Expose a basic /metrics endpoint (Optional)
@app.get("/metrics")
def get_metrics():
    # In a real-world scenario, you would expose metrics from Prometheus here
    return {"message": "Metrics endpoint (not fully implemented for this demo)"}


# --- Main entry point to run the application ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
  
