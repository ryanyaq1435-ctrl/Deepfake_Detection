import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from model import DeepFakeDetector

# =========================
# CONFIG (MATCH TRAINING)
# =========================
SEQUENCE_LENGTH = 10   # MUST match training
IMAGE_SIZE = 112

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = DeepFakeDetector()
    model.load_state_dict(torch.load("models/best_model_epoch_7.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# =========================
# FACE EXTRACTION
# =========================
def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(min_detection_confidence=0.5)

    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # ✅ SAFE BOUNDARIES FIX
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            face = frame[y:y+height, x:x+width]

            if face.size > 0:
                face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                face = face / 255.0
                frames.append(face)

    cap.release()

    # Padding if fewer frames
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))

    frames = np.array(frames)
    frames = np.transpose(frames, (0, 3, 1, 2))

    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Deepfake Detector", page_icon="🎭")

st.title("🎭 Deepfake Video Detector")
st.write("Upload a video to check if it's real or AI-generated")

uploaded_file = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    if st.button("🔍 Detect"):
        with st.spinner("Analyzing video..."):

            x = extract_faces(tfile.name)

            with torch.no_grad():
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)

                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()

            label = "REAL ✅" if pred == 0 else "FAKE ❌"

            st.subheader(label)
            st.write(f"Confidence: {conf:.2%}")

            st.progress(float(conf))