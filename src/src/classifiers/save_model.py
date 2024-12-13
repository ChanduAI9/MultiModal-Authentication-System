import joblib
import os

def save_model(model, scaler, filename):
    """
    Save a trained model and scaler to a file.

    Parameters:
    - model: Trained classifier (e.g., kNN, SVM).
    - scaler: Preprocessing scaler used during training.
    - filename: Path to save the model (e.g., "outputs/models/knn_model.pkl").
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the model and scaler
    data = {"model": model, "scaler": scaler}
    joblib.dump(data, filename)
    print(f"Model and scaler saved to: {filename}")
