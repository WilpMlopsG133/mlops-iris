# app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def load_model(model_name):
    """Load model from local models/ directory."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON:
    {
        "model_name": "LogisticRegression",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    data = request.get_json()

    if not data or "model_name" not in data or "features" not in data:
        return jsonify({"error": "Missing model_name or features"}), 400

    model_name = data["model_name"]
    features = data["features"]

    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"Model {model_name} not found in {MODEL_DIR}"}), 404

    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify({"model": model_name, "prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


