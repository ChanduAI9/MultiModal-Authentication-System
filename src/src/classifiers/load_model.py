import joblib

def load_model(filename):
    """
    Load a trained model and scaler from a file.

    Parameters:
    - filename: Path to the saved model file (e.g., "outputs/models/knn_model.pkl").

    Returns:
    - model: Loaded classifier.
    - scaler: Loaded preprocessing scaler.
    """
    data = joblib.load(filename)
    print(f"Model and scaler loaded from: {filename}")
    return data["model"], data["scaler"]
