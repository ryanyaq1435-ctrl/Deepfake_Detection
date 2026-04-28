"""
Deepfake Detection - Preprocessing Script
Handles face extraction from videos
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ✅ FIXED MediaPipe import (stable)
from mediapipe.python.solutions import face_detection as mp_face_detection

# =========================
# CHECK DEPENDENCIES
# =========================
def check_dependencies():
    try:
        import cv2
        import mediapipe
        import numpy
        print("✅ All dependencies satisfied!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

# =========================
# SETUP DIRECTORIES
# =========================
def setup_directories():
    directories = [
        "data/raw/train/real",
        "data/raw/train/fake",
        "data/raw/test/real",
        "data/raw/test/fake",
        "data/processed/train_real_faces",
        "data/processed/train_fake_faces",
        "data/processed/test_real_faces",
        "data/processed/test_fake_faces",
        "models"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")

# =========================
# FACE EXTRACTION FUNCTION
# =========================
def extract_faces_from_video(video_path, output_path, target_size=(112, 112), max_frames=150):

    try:
        # Initialize face detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Cannot open: {video_path}")
            return 0

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

        frame_count = 0
        faces_extracted = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                for detection in results.detections:

                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Padding
                    padding = 0.1
                    x = max(0, int(x - padding * width))
                    y = max(0, int(y - padding * height))
                    width = int(width * (1 + 2 * padding))
                    height = int(height * (1 + 2 * padding))

                    # Boundary check
                    x = min(x, w - 1)
                    y = min(y, h - 1)
                    width = min(width, w - x)
                    height = min(height, h - y)

                    face = frame[y:y+height, x:x+width]

                    if face.size > 0:
                        face = cv2.resize(face, target_size)

                        # ✅ NORMALIZATION (IMPORTANT)
                        face = face / 255.0

                        # Convert back to uint8 for saving video
                        face_uint8 = (face * 255).astype(np.uint8)

                        out.write(face_uint8)
                        faces_extracted += 1

            frame_count += 1

            if frame_count % 50 == 0:
                print(f"  Processed {frame_count} frames...")

        cap.release()
        out.release()
        face_detector.close()

        return faces_extracted

    except Exception as e:
        print(f"❌ Error: {video_path} → {e}")
        return 0

# =========================
# PROCESS DATASET
# =========================
def process_dataset():

    print("\n📹 Starting preprocessing...")

    raw_base = Path("data/raw")
    processed_base = Path("data/processed")

    if not raw_base.exists():
        print("❌ Raw dataset not found!")
        return

    splits = ['train', 'test']
    labels = ['real', 'fake']

    total_videos = 0
    total_faces = 0

    for split in splits:
        for label in labels:

            input_dir = raw_base / split / label
            output_dir = processed_base / f"{split}_{label}_faces"

            if not input_dir.exists():
                continue

            videos = list(input_dir.glob("*.mp4"))

            print(f"\n📂 {split}/{label} → {len(videos)} videos")

            for video in videos:
                output_file = output_dir / f"{video.stem}_faces.mp4"

                if output_file.exists():
                    print(f"⏭️ Skipping: {video.name}")
                    continue

                print(f"🎬 Processing: {video.name}")
                count = extract_faces_from_video(str(video), str(output_file))

                if count > 0:
                    print(f"   ✅ {count} faces extracted")
                    total_faces += count
                    total_videos += 1
                else:
                    print("   ❌ No faces found")

    print("\n✅ DONE!")
    print(f"Videos: {total_videos}")
    print(f"Faces: {total_faces}")

# =========================
# SAMPLE VIDEO (OPTIONAL)
# =========================
def create_sample_video():
    print("\n📹 Creating sample video...")

    path = Path("data/raw/train/real/sample.mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    for _ in range(150):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.circle(frame, (320, 240), 100, (255, 200, 150), -1)
        cv2.circle(frame, (280, 210), 15, (0, 0, 0), -1)
        cv2.circle(frame, (360, 210), 15, (0, 0, 0), -1)
        cv2.ellipse(frame, (320, 280), (40, 25), 0, 0, 180, (100, 50, 50), -1)

        out.write(frame)

    out.release()
    print(f"✅ Sample video created: {path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\n🎭 Deepfake Detection - Preprocessing\n")

    if not check_dependencies():
        sys.exit(1)

    setup_directories()

    choice = input("Create sample video? (y/n): ")
    if choice.lower() == 'y':
        create_sample_video()

    process_dataset()

    print("\n🚀 Next Step: Run Training_Loop.py")