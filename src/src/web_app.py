import streamlit as st
from classifiers.load_model import load_model
from data.get_landmarks import get_landmarks
from data.get_palmprint_features import get_palmprint_features
from fusion.score_level import score_level_fusion
import numpy as np

st.title("Multimodal Authentication System")

uploaded_face = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
uploaded_palm = st.file_uploader("Upload Palmprint Image", type=["jpg", "png", "jpeg"])

if uploaded_face and uploaded_palm:
    # Process images
    face_img = cv2.imdecode(np.frombuffer(uploaded_face.read(), np.uint8), cv2.IMREAD_COLOR)
    palm_img = cv2.imdecode(np.frombuffer(uploaded_palm.read(), np.uint8), cv2.IMREAD_COLOR)

    face_landmarks = get_landmarks([face_img], ["test"])
    palm_features = get_palmprint_features([palm_img])

    if face_landmarks and palm_features:
        face_clf, _ = load_model("outputs/models/face_knn.pkl")
        palm_clf, _ = load_model("outputs/models/palm_svm.pkl")

        face_prob = face_clf.predict_proba([face_landmarks[0]])
        palm_prob = palm_clf.predict_proba([palm_features[0]])

        fused_prob = score_level_fusion(face_prob[0], palm_prob[0])
        prediction = np.argmax(fused_prob)

        st.write(f"Predicted Label: {prediction}")
    else:
        st.write("Could not extract features from both images.")
