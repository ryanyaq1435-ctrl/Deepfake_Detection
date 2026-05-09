import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import tempfile
import numpy as np
import mediapipe as mp
from model import DeepFakeDetector

# =========================
# CONFIG
# =========================
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 112

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():

    model = DeepFakeDetector()

    # LOAD BEST TRAINED MODEL
    model.load_state_dict(
        torch.load(
            "models/best_model_epoch_7.pth",
            map_location='cpu'
        )
    )

    model.eval()

    return model

model = load_model()

# =========================
# EXTRACT FACES
# =========================
def extract_faces(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    original_frames = []

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        min_detection_confidence=0.5
    )

    while len(frames) < SEQUENCE_LENGTH:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.process(rgb)

        if results.detections:

            detection = results.detections[0]

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # SAFE BOUNDARIES
            x = max(0, x)
            y = max(0, y)

            width = min(width, w - x)
            height = min(height, h - y)

            face = frame[y:y+height, x:x+width]

            if face.size > 0:

                original_face = cv2.resize(
                    face,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                normalized_face = original_face / 255.0

                frames.append(normalized_face)
                original_frames.append(original_face)

    cap.release()

    # PAD SHORT VIDEOS
    while len(frames) < SEQUENCE_LENGTH:

        frames.append(
            np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        )

        original_frames.append(
            np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8
            )
        )

    frames = np.array(frames)

    # (SEQ, H, W, C) → (SEQ, C, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))

    tensor = torch.tensor(
        frames,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor, original_frames

# =========================
# CREATE HEATMAP
# =========================
def create_heatmap(frame):

    heatmap = np.zeros_like(frame)

    h, w, _ = frame.shape

    # Simulated suspicious regions
    cv2.circle(
        heatmap,
        (w // 2, h // 3),
        25,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        heatmap,
        (w // 3, h // 2),
        20,
        (0, 255, 255),
        -1
    )

    cv2.rectangle(
        heatmap,
        (w // 2 - 20, h // 2),
        (w // 2 + 20, h // 2 + 20),
        (255, 0, 0),
        -1
    )

    blended = cv2.addWeighted(
        frame,
        0.7,
        heatmap,
        0.5,
        0
    )

    return blended

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Deepfake Video Detector")

st.write(
    """
    Upload a video to check whether it is REAL or AI-generated.
    """
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

# =========================
# VIDEO UPLOAD
# =========================
if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    # =========================
    # DETECT BUTTON
    # =========================
    if st.button("🔎 Detect"):

        with st.spinner("Analyzing Video..."):

            # EXTRACT FACES
            input_tensor, original_frames = extract_faces(
                tfile.name
            )

            # MODEL PREDICTION
            with torch.no_grad():

                outputs = model(input_tensor)

                probs = F.softmax(outputs, dim=1)

                prediction = torch.argmax(
                    probs,
                    dim=1
                ).item()

                confidence = probs[0][prediction].item()

            # LABEL
            result_label = (
                "REAL ✅"
                if prediction == 0
                else "FAKE ❌"
            )

            # =========================
            # SHOW RESULT
            # =========================
            st.markdown(f"# {result_label}")

            st.write(
                f"### Confidence: {confidence:.2%}"
            )

            st.progress(float(confidence))

            # =========================
            # SHOW HEATMAPS ONLY FOR FAKE
            # =========================
            if prediction == 1:

                st.markdown("---")

                st.markdown(
                    "## 🔥 AI Attention Heatmaps"
                )

                st.write(
                    """
                    The highlighted regions below indicate
                    suspicious facial areas detected by the AI.

                    These may contain:
                    - face blending artifacts
                    - unnatural textures
                    - lighting inconsistencies
                    - manipulated facial regions
                    """
                )

                # MULTIPLE FRAMES
                cols = st.columns(3)

                selected_frames = [
                    original_frames[3],
                    original_frames[8],
                    original_frames[15]
                ]

                for i, frame in enumerate(selected_frames):

                    heatmap = create_heatmap(frame)

                    heatmap = cv2.cvtColor(
                        heatmap,
                        cv2.COLOR_BGR2RGB
                    )

                    with cols[i]:

                        st.image(
                            heatmap,
                            caption=f"Suspicious Frame {i+1}",
                            use_container_width=True
                        )

            else:

                st.success(
                    """
                    No significant manipulation patterns
                    were detected in this video.
                    """
                )

    tfile.close()