# predict.py
import sys
import joblib
import numpy as np

def load_model(model_name):
    """Load model from the local 'models/' directory."""
    model_path = f"models/{model_name}.joblib"
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

def predict(model_name, features):
    """Predict class for given features."""
    model = load_model(model_name)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]

if __name__ == "__main__":
    # Example usage: python predict.py LogisticRegression 5.1 3.5 1.4 0.2
    if len(sys.argv) < 6:
        print("Usage: python predict.py <model_name> <feature1> <feature2> <feature3> <feature4>")
        sys.exit(1)

    model_name = sys.argv[1]
    features = [float(x) for x in sys.argv[2:6]]
    result = predict(model_name, features)
    print(f"Predicted class: {result}")

