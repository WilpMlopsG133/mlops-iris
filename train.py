# train.py
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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


