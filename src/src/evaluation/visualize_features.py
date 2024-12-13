import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_feature_space(features, labels, scaler=None, filename=None):
    """
    Visualize the feature space and save it as an image.

    Parameters:
    - features: Feature matrix (list of features).
    - labels: Corresponding labels.
    - scaler: Scaler to normalize features (Optional).
    - filename: Filename to save the visualization (Optional).
    """
    # Convert features to NumPy arrays and ensure consistent shape
    features = [np.array(f) for f in features]
    max_length = max(len(f.flatten()) for f in features)
    consistent_features = np.array([np.pad(f.flatten(), (0, max_length - len(f.flatten()))) for f in features])

    # Ensure the number of labels matches the number of features
    if len(labels) != consistent_features.shape[0]:
        raise ValueError(f"Number of labels ({len(labels)}) does not match the number of features ({consistent_features.shape[0]}).")

    # Scale features if a scaler is provided
    if scaler:
        consistent_features = scaler.transform(consistent_features)

    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(consistent_features)

    # Map string labels to numeric values
    unique_labels = sorted(set(labels))
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in labels])

    # Plot the reduced features
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=numeric_labels, cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="Class Label")
    plt.title("Feature Space Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Feature space visualization saved as {filename}")

    plt.show()
