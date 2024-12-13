from imblearn.over_sampling import SMOTE

def oversample_data(X, y):
    """
    Oversample data using SMOTE.

    Parameters:
    - X: Feature matrix.
    - y: Labels.

    Returns:
    - X_resampled: Resampled feature matrix.
    - y_resampled: Resampled labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Data resampled. Original size: {len(y)}, Resampled size: {len(y_resampled)}")
    return X_resampled, y_resampled
