import cv2
import numpy as np

def get_palmprint_features(images):
    """
    Extract features from palmprint images.

    Parameters:
    - images: List of images.

    Returns:
    - List of feature vectors.
    """
    features = []

    for idx, image in enumerate(images):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            features.append(resized.flatten())
        except Exception as e:
            print(f"Error processing palmprint image at index {idx}: {e}")

    print(f"Extracted features for {len(features)} palmprint images.")
    return features
