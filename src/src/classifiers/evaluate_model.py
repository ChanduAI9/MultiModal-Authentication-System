from sklearn.model_selection import cross_val_score

def evaluate_cross_validation(model, X_train, y_train, model_name="Model"):
    """
    Perform k-fold cross-validation to evaluate the model.

    Parameters:
    - model: Classifier (e.g., kNN, SVM).
    - X_train: Training features.
    - y_train: Training labels.
    - model_name: Name of the model for logging purposes.

    Returns:
    - scores: Cross-validation scores.
    """
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{model_name} Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean()}")
    return scores
