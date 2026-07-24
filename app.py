import cv2
import gradio as gr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import time
import urllib.request

# ==========================================
# 1. STUDIO ACOUSTIC PIANO SOUND ENGINE
# ==========================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

WHITE_KEYS = [
    {"name": "C4", "file": "C4.mp3", "freq": 261.63},
    {"name": "D4", "file": "D4.mp3", "freq": 293.66},
    {"name": "E4", "file": "E4.mp3", "freq": 329.63},
    {"name": "F4", "file": "F4.mp3", "freq": 349.23},
    {"name": "G4", "file": "G4.mp3", "freq": 392.00},
    {"name": "A4", "file": "A4.mp3", "freq": 440.00},
    {"name": "B4", "file": "B4.mp3", "freq": 493.88},
    {"name": "C5", "file": "C5.mp3", "freq": 523.25},
    {"name": "D5", "file": "D5.mp3", "freq": 587.33},
    {"name": "E5", "file": "E5.mp3", "freq": 659.25},
]

BLACK_KEYS = [
    {"name": "C#4", "file": "Db4.mp3", "freq": 277.18, "after_white": 0},
    {"name": "D#4", "file": "Eb4.mp3", "freq": 311.13, "after_white": 1},
    {"name": "F#4", "file": "Gb4.mp3", "freq": 369.99, "after_white": 3},
    {"name": "G#4", "file": "Ab4.mp3", "freq": 415.30, "after_white": 4},
    {"name": "A#4", "file": "Bb4.mp3", "freq": 466.16, "after_white": 5},
    {"name": "C#5", "file": "Db5.mp3", "freq": 554.37, "after_white": 7},
    {"name": "D#5", "file": "Eb5.mp3", "freq": 622.25, "after_white": 8},
]

SOUND_DIR = "sounds"
os.makedirs(SOUND_DIR, exist_ok=True)
BASE_SOUND_URL = "https://raw.githubusercontent.com/gleitz/midi-js-soundfonts/gh-pages/FluidR3_GM/acoustic_grand_piano-mp3/"

SOUND_FILE_MAP = {}
for k in WHITE_KEYS + BLACK_KEYS:
    local_path = os.path.join(SOUND_DIR, k["file"])
    if not os.path.exists(local_path):
        try:
            urllib.request.urlretrieve(BASE_SOUND_URL + k["file"], local_path)
        except Exception:
            pass
    if os.path.exists(local_path):
        SOUND_FILE_MAP[k["name"]] = local_path

class WebTracker:
    def __init__(self):
        self.active_key = None

tracker = WebTracker()

def process_webcam_frame(frame):
    if frame is None:
        return None, None, "Waiting for webcam feed..."

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    key_area_top = int(h * 0.55)
    key_area_height = h - key_area_top
    num_white = len(WHITE_KEYS)
    white_width = w // num_white
    black_width = int(white_width * 0.6)
    black_height = int(key_area_height * 0.6)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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

    audio_output = None
    note_status = "Ready — Move index finger over piano keys!"

    if detected_key_name:
        note_status = f"Playing Note: 🎹 {detected_key_name}"
        if detected_key_name != tracker.active_key:
            tracker.active_key = detected_key_name
            audio_output = SOUND_FILE_MAP.get(detected_key_name)
    else:
        tracker.active_key = None

    # Render Graphics
    overlay = frame.copy()

    # White Keys
    for i, wk in enumerate(WHITE_KEYS):
        x1 = i * white_width
        x2 = (i + 1) * white_width if i < num_white - 1 else w
        y1 = key_area_top
        y2 = h
        is_pressed = (wk["name"] == tracker.active_key)
        
        if is_pressed:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 150), -1)
            cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 200, 100), 3)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (240, 240, 240), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 2)

        lbl = wk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        txt_color = (0, 0, 0) if is_pressed else (40, 40, 40)
        cv2.putText(frame, lbl, (x1 + (white_width - tw) // 2, y2 - 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_color, 2)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    overlay = frame.copy()

    # Black Keys
    for bk in BLACK_KEYS:
        idx = bk["after_white"]
        bx1 = int((idx + 1) * white_width - (black_width / 2))
        bx2 = bx1 + black_width
        by1 = key_area_top
        by2 = key_area_top + black_height
        is_pressed = (bk["name"] == tracker.active_key)

        if is_pressed:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 220, 255), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), 2)

        lbl = bk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        txt_col = (0, 0, 0) if is_pressed else (220, 220, 220)
        cv2.putText(frame, lbl, (bx1 + (black_width - tw) // 2, by2 - 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, txt_col, 1)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Finger HUD Reticle
    if current_finger_pt:
        fx, fy = current_finger_pt
        cv2.circle(frame, (fx, fy), 16, (0, 255, 255), 2)
        cv2.circle(frame, (fx, fy), 6, (0, 165, 255), -1)
        cv2.line(frame, (fx - 24, fy), (fx + 24, fy), (0, 255, 255), 1)
        cv2.line(frame, (fx, fy - 24), (fx, fy + 24), (0, 255, 255), 1)

    # HUD Banner
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 65), (15, 15, 25), -1)
    cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "AIR PIANO PRO", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Studio Acoustic Piano — Steven Thomas", (20, 52), cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 180), 1)

    return frame, audio_output, note_status

# ==========================================
# 2. GRADIO WEB INTERFACE
# ==========================================
with gr.Blocks(title="Air Piano Pro by Steven Thomas", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎹 Air Piano Pro
        ### Created by **Steven Thomas**
        High-fidelity Studio Acoustic Grand Piano with computer vision hand tracking!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
        with gr.Column(scale=1):
            output_status = gr.Textbox(label="Status & Active Note", value="Initializing hand tracker...")
            output_audio = gr.Audio(label="Audio Output", autoplay=True)

    input_image.stream(
        fn=process_webcam_frame,
        inputs=[input_image],
        outputs=[input_image, output_audio, output_status],
        stream_every=0.1
    )

if __name__ == "__main__":
    demo.launch()
