from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_knn(landmarks, labels):
    """
    Train a k-Nearest Neighbors (kNN) classifier.

    Parameters:
    - landmarks: List of landmark features extracted from images.
    - labels: Corresponding labels for the images.

    Returns:
    - clf: Trained kNN classifier.
    - scaler: Scaler used for feature normalization.
    - X_test: Test features (for evaluation).
    - y_test: Test labels (for evaluation).
    - y_pred: Predicted labels for the test set.
    """
    import numpy as np

    # Flatten landmarks for input into the model
    X = np.array([np.array(landmark).flatten() for landmark in landmarks])
    y = np.array(labels)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Train the kNN classifier
    clf = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"kNN Classifier Accuracy: {accuracy * 100:.2f}%")

    return clf, scaler, X_test, y_test, y_pred
