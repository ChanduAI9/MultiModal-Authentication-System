import mediapipe as mp
import numpy as np

def get_landmarks(images, labels):
    """
    Extract facial landmarks for each image.

    Parameters:
    - images: List of images.
    - labels: List of corresponding labels.

    Returns:
    - List of extracted landmarks.
    """
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    landmarks = []

    for idx, image in enumerate(images):
        try:
            results = mp_face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_data = [
                        [landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark
                    ]
                    landmarks.append(np.array(face_data).flatten())
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")

    print(f"Extracted landmarks for {len(landmarks)} images.")
    return landmarks
