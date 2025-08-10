<<<<<<< HEAD
# train.py
import os
import joblib
import mlflow
import mlflow.sklearn
=======
# mlops-iris/app/train.py

import os
import pandas as pd
>>>>>>> b73708408ebc4425ddcb5d490be296ebce87a24a
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
<<<<<<< HEAD

# Ensure local models directory exists
os.makedirs("models", exist_ok=True)

mlflow.set_experiment("Default")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("LogisticRegression", LogisticRegression(max_iter=500)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42))
]

for name, clf in models:
    with mlflow.start_run(run_name=name) as run:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params and metrics to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", float(acc))

        # Log model to MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # Save a local copy in 'models/' directory
        model_path = f"models/{name}.joblib"
        joblib.dump(clf, model_path)
        print(f"Local model saved to {model_path}")

        print(f"Logged run: name={name} run_id={run.info.run_id} accuracy={acc:.4f}")
=======
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

def train_and_register_models():
    """Trains multiple models and registers the best one with MLflow."""
    print("Loading and preprocessing data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Use a simple relative path for the tracking URI.
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Iris_Model_Training")

    best_accuracy = 0
    best_model_uri = ""

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

        print(f"Logistic Regression Accuracy: {accuracy}")

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

        print(f"Random Forest Accuracy: {accuracy}")

    print("Registering the best model...")
    model_details = mlflow.register_model(best_model_uri, "IrisBestModel")
    print(f"Best model registered from: {best_model_uri}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name="IrisBestModel",
        version=model_details.version,
        stage="Production"
    )
    print(f"Version {model_details.version} of model 'IrisBestModel' has been transitioned to the 'Production' stage.")
    print(f"Best model registered from: {best_model_uri}")

if __name__ == "__main__":
    train_and_register_models()
>>>>>>> b73708408ebc4425ddcb5d490be296ebce87a24a


