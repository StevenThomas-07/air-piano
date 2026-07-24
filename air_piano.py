import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pygame
import sys
import time
import urllib.request

# ==========================================
# 1. STUDIO ACOUSTIC PIANO SOUND ENGINE
# ==========================================
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Piano Keys Definition (White & Black Keys)
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

def generate_fallback_piano_tone(freq, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    fundamental = np.sin(2 * np.pi * freq * t)
    h2 = 0.45 * np.sin(2 * np.pi * 2 * freq * t)
    h3 = 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    raw_wave = fundamental + h2 + h3
    envelope = np.exp(-3.2 * t)
    signal = raw_wave * envelope * 0.7
    mono = (signal * 32767).astype(np.int16)
    stereo = np.column_stack((mono, mono))
    return pygame.sndarray.make_sound(stereo)

SOUND_MAP = {}
print("Loading Studio Acoustic Grand Piano samples...")

for k in WHITE_KEYS + BLACK_KEYS:
    local_path = os.path.join(SOUND_DIR, k["file"])
    if not os.path.exists(local_path):
        url = BASE_SOUND_URL + k["file"]
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Warning: Could not download {k['file']}, using synthesized fallback.")

    if os.path.exists(local_path):
        try:
            SOUND_MAP[k["name"]] = pygame.mixer.Sound(local_path)
        except Exception:
            SOUND_MAP[k["name"]] = generate_fallback_piano_tone(k["freq"])
    else:
        SOUND_MAP[k["name"]] = generate_fallback_piano_tone(k["freq"])

print("Acoustic Grand Piano loaded successfully!")


# ==========================================
# 2. MEDIAPIPE MODEL SETUP
# ==========================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ==========================================
# 3. MAIN LOOP & RENDERING
# ==========================================
active_key = None
trail_points = []
wave_visualizer_offset = 0

print("Starting AIR PIANO PRO by Steven Thomas...")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    key_area_top = int(h * 0.55)
    key_area_height = h - key_area_top
    num_white = len(WHITE_KEYS)
    white_width = w // num_white
    black_width = int(white_width * 0.6)
    black_height = int(key_area_height * 0.6)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_finger_pt = None
    detected_key_name = None

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            current_finger_pt = (cx, cy)
            
            trail_points.append({'pt': (cx, cy), 'time': time.time()})

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

    now = time.time()
    trail_points = [p for p in trail_points if now - p['time'] < 0.4]

    # Debouncing sound trigger
    if detected_key_name:
        if detected_key_name != active_key:
            SOUND_MAP[detected_key_name].stop()  # stop previous instance if playing
            SOUND_MAP[detected_key_name].play()
            active_key = detected_key_name
    else:
        active_key = None

    # ==========================================
    # 4. RENDER GRAPHICS & HUD OVERLAY
    # ==========================================
    overlay = frame.copy()

    # Draw White Keys
    for i, wk in enumerate(WHITE_KEYS):
        x1 = i * white_width
        x2 = (i + 1) * white_width if i < num_white - 1 else w
        y1 = key_area_top
        y2 = h

        is_pressed = (wk["name"] == active_key)
        
        if is_pressed:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 150), -1)
            cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 200, 100), 4)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (240, 240, 240), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 2)

        txt_color = (0, 0, 0) if is_pressed else (40, 40, 40)
        lbl = wk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(frame, lbl, (x1 + (white_width - tw) // 2, y2 - 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, txt_color, 2)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    overlay = frame.copy()

    # Draw Black Keys
    for bk in BLACK_KEYS:
        idx = bk["after_white"]
        bx1 = int((idx + 1) * white_width - (black_width / 2))
        bx2 = bx1 + black_width
        by1 = key_area_top
        by2 = key_area_top + black_height

        is_pressed = (bk["name"] == active_key)

        if is_pressed:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 220, 255), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), 2)

        lbl = bk["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        txt_col = (0, 0, 0) if is_pressed else (220, 220, 220)
        cv2.putText(frame, lbl, (bx1 + (black_width - tw) // 2, by2 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.55, txt_col, 1)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Particle Trails
    for i in range(1, len(trail_points)):
        pt1 = trail_points[i - 1]['pt']
        pt2 = trail_points[i]['pt']
        alpha = (i / len(trail_points))
        thickness = int(alpha * 8) + 1
        cv2.line(frame, pt1, pt2, (0, int(255 * alpha), 255), thickness)

    # Reticle HUD
    if current_finger_pt:
        fx, fy = current_finger_pt
        cv2.circle(frame, (fx, fy), 16, (0, 255, 255), 2)
        cv2.circle(frame, (fx, fy), 6, (0, 165, 255), -1)
        cv2.line(frame, (fx - 24, fy), (fx + 24, fy), (0, 255, 255), 1)
        cv2.line(frame, (fx, fy - 24), (fx, fy + 24), (0, 255, 255), 1)

    # Top HUD Banner
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 75), (15, 15, 25), -1)
    cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "AIR PIANO PRO", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, "Acoustic Studio Sound — Steven Thomas", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.45, (180, 180, 180), 1)

    # Active Note Display
    if active_key:
        note_str = f"ACTIVE NOTE: {active_key}"
        cv2.rectangle(frame, (w // 2 - 140, 15), (w // 2 + 140, 60), (0, 255, 120), -1)
        (nw, nh), _ = cv2.getTextSize(note_str, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.putText(frame, note_str, (w // 2 - nw // 2, 43), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    else:
        note_str = "READY - MOVE FINGER TO PLAY"
        (nw, nh), _ = cv2.getTextSize(note_str, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.putText(frame, note_str, (w // 2 - nw // 2, 43), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1)

    # Equalizer Visualizer
    wave_visualizer_offset += 0.2
    for bar_i in range(12):
        bar_x = w - 180 + (bar_i * 12)
        if active_key:
            bar_h = int(np.abs(np.sin(wave_visualizer_offset + bar_i * 0.5)) * 35) + 8
            bar_color = (0, 255, 150)
        else:
            bar_h = int(np.abs(np.sin(wave_visualizer_offset * 0.5 + bar_i * 0.3)) * 8) + 4
            bar_color = (100, 100, 100)
        cv2.rectangle(frame, (bar_x, 60 - bar_h), (bar_x + 8, 60), bar_color, -1)

    cv2.imshow("AIR PIANO PRO - Steven Thomas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
