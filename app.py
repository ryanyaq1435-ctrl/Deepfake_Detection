import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
from model import DeepFakeDetector

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 112

# =========================
# FUNCTION: PROCESS VIDEO
# =========================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Padding if video is short
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))

    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (seq, C, H, W)

    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Deepfake Detector", page_icon="🎭")

st.title("🎭 Deepfake Video Detector")
st.write("Upload a video to check if it's real or AI-generated")

uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    if st.button("🔍 Detect"):
        with st.spinner("Analyzing video..."):

            # Load model
            model = DeepFakeDetector()
            model.load_state_dict(torch.load("models/best_model_epoch_1.pth", map_location='cpu'))
            model.eval()

            # Process video
            input_tensor = extract_frames(tfile.name)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)

                confidence = probs[0][1].item()
                prediction = torch.argmax(probs, dim=1).item()

            # Label
            result_label = "FAKE ❌" if prediction == 1 else "REAL ✅"

            # Display
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", result_label)

            with col2:
                st.metric("Confidence", f"{confidence:.2%}")

            st.progress(float(confidence))

    tfile.close()