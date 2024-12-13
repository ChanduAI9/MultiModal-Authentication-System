import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from data.get_images import get_images
from data.get_landmarks import get_landmarks
from data.get_palmprint_features import get_palmprint_features
from fusion.score_level import score_level_fusion
from classifiers.save_model import save_model
from evaluation.performance_plots import plot_confusion_matrix

def main():
    face_dir = r"C:\Users\HP\Documents\Chandu\python_project\Code\multimodal_authentication\multimodal_authentication\datasets\datasets\face"
    palmprint_dir = r"C:\Users\HP\Documents\Chandu\python_project\Code\multimodal_authentication\multimodal_authentication\datasets\datasets\palmprint"

    # Load datasets
    face_images, face_labels = get_images(face_dir)
    palm_images, palm_labels = get_images(palmprint_dir)

    # Ensure labels match across modalities
    if set(face_labels) != set(palm_labels):
        raise ValueError("Labels for face and palmprint datasets do not match!")

    # Extract features
    face_landmarks = get_landmarks(face_images, face_labels)
    palm_features = get_palmprint_features(palm_images)

    # Ensure features and labels align
    if len(face_landmarks) != len(palm_features):
        raise ValueError("Mismatch between face and palmprint feature counts!")

    # Split data
    labels = face_labels  # Common labels
    X_face_train, X_face_test, y_train, y_test = train_test_split(face_landmarks, labels, test_size=0.2, random_state=42)
    X_palm_train, X_palm_test, _, _ = train_test_split(palm_features, labels, test_size=0.2, random_state=42)

    # Train classifiers
    face_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    face_clf.fit(X_face_train, y_train)
    save_model(face_clf, None, "outputs/models/face_knn.pkl")

    palm_clf = SVC(kernel="linear", probability=True, C=1.0)
    palm_clf.fit(X_palm_train, y_train)
    save_model(palm_clf, None, "outputs/models/palm_svm.pkl")

    # Predict and fuse
    face_probs = face_clf.predict_proba(X_face_test)
    palm_probs = palm_clf.predict_proba(X_palm_test)
    fused_probs = np.array([score_level_fusion(f, p) for f, p in zip(face_probs, palm_probs)])
    fused_preds = np.argmax(fused_probs, axis=1)

    # Evaluate
    plot_confusion_matrix(y_test, fused_preds, classes=np.unique(labels), title="Fusion Confusion Matrix")
    print("Multimodal Authentication System Complete.")

if __name__ == "__main__":
    main()
