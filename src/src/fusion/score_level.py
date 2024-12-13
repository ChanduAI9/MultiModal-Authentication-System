import numpy as np

def score_level_fusion(face_prob, palmprint_prob, alpha=0.5):
    """
    Combine face and palmprint probabilities using score-level fusion.

    Parameters:
    - face_prob: Prediction probabilities from face classifier.
    - palmprint_prob: Prediction probabilities from palmprint classifier.
    - alpha: Weight for face probabilities.

    Returns:
    - fused_prob: Combined probabilities.
    """
    fused_prob = alpha * face_prob + (1 - alpha) * palmprint_prob
    return fused_prob
