import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import streamlit as st
import urllib.request

# Page configuration
st.set_page_config(
    page_title="Air Piano Pro - Steven Thomas",
    page_icon="🎹",
    layout="centered"
)

st.title("🎹 Air Piano Pro")
st.markdown("### Created by **Steven Thomas**")
st.write("Point your camera at your hand and move your index finger over the piano keys!")

# Download MediaPipe model
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading hand tracking AI model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    return vision.HandLandmarker.create_from_options(options)

detector = load_detector()

WHITE_KEYS = [
    {"name": "C4", "freq": 261.63},
    {"name": "D4", "freq": 293.66},
    {"name": "E4", "freq": 329.63},
    {"name": "F4", "freq": 349.23},
    {"name": "G4", "freq": 392.00},
    {"name": "A4", "freq": 440.00},
    {"name": "B4", "freq": 493.88},
    {"name": "C5", "freq": 523.25},
    {"name": "D5", "freq": 587.33},
    {"name": "E5", "freq": 659.25},
]

BLACK_KEYS = [
    {"name": "C#4", "freq": 277.18, "after_white": 0},
    {"name": "D#4", "freq": 311.13, "after_white": 1},
    {"name": "F#4", "freq": 369.99, "after_white": 3},
    {"name": "G#4", "freq": 415.30, "after_white": 4},
    {"name": "A#4", "freq": 466.16, "after_white": 5},
    {"name": "C#5", "freq": 554.37, "after_white": 7},
    {"name": "D#5", "freq": 622.25, "after_white": 8},
]

# Camera Input
img_file_buffer = st.camera_input("Take a snapshot or turn on camera")

if img_file_buffer is not None:
    # Convert image buffer to OpenCV BGR
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    cv2_img = cv2.flip(cv2_img, 1)
    h, w, _ = cv2_img.shape

    key_area_top = int(h * 0.55)
    key_area_height = h - key_area_top
    num_white = len(WHITE_KEYS)
    white_width = w // num_white
    black_width = int(white_width * 0.6)
    black_height = int(key_area_height * 0.6)

    rgb_frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_finger_pt = None
    detected_key_name = None

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            current_finger_pt = (cx, cy)

            if cy >= key_area_top:
                in_black = False
                for bk in BLACK_KEYS:
                    idx = bk["after_white"]
                    bx1 = int((idx + 1) * white_width - (black_width / 2))
                    bx2 = bx1 + black_width
                    by1 = key_area_top
                    by2 = key_area_top + black_height
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        detected_key_name = bk["name"]
                        in_black = True
                        break

                if not in_black:
                    w_idx = min(max(cx // white_width, 0), num_white - 1)
                    detected_key_name = WHITE_KEYS[w_idx]["name"]

    # Render Graphics Overlay
    overlay = cv2_img.copy()

    for i, wk in enumerate(WHITE_KEYS):
        x1 = i * white_width
        x2 = (i + 1) * white_width if i < num_white - 1 else w
        y1 = key_area_top
        y2 = h
        is_pressed = (wk["name"] == detected_key_name)
        
        if is_pressed:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 150), -1)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (240, 240, 240), -1)
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (80, 80, 80), 2)

        lbl = wk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        txt_color = (0, 0, 0) if is_pressed else (40, 40, 40)
        cv2.putText(cv2_img, lbl, (x1 + (white_width - tw) // 2, y2 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_color, 2)

    cv2.addWeighted(overlay, 0.4, cv2_img, 0.6, 0, cv2_img)
    overlay = cv2_img.copy()

    for bk in BLACK_KEYS:
        idx = bk["after_white"]
        bx1 = int((idx + 1) * white_width - (black_width / 2))
        bx2 = bx1 + black_width
        by1 = key_area_top
        by2 = key_area_top + black_height
        is_pressed = (bk["name"] == detected_key_name)

        if is_pressed:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 220, 255), -1)
        else:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(cv2_img, (bx1, by1), (bx2, by2), (60, 60, 60), 2)

        lbl = bk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        txt_col = (0, 0, 0) if is_pressed else (220, 220, 220)
        cv2.putText(cv2_img, lbl, (bx1 + (black_width - tw) // 2, by2 - 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, txt_col, 1)

    cv2.addWeighted(overlay, 0.7, cv2_img, 0.3, 0, cv2_img)

    if current_finger_pt:
        fx, fy = current_finger_pt
        cv2.circle(cv2_img, (fx, fy), 16, (0, 255, 255), 2)
        cv2.circle(cv2_img, (fx, fy), 6, (0, 165, 255), -1)

    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    if detected_key_name:
        st.success(f"Playing Note: 🎹 **{detected_key_name}**")
    else:
        st.info("Move your index finger over the piano keys!")
