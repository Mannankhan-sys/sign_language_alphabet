"""
ALPHABET DETECTOR - ULTIMATE VERSION
======================================
Advanced real-time alphabet sign language detector with:
✓ Stable detection voting system (10 consecutive predictions)
✓ Hand positioning guide box with visual feedback
✓ HD window (1280x720)
✓ A-Z letters + SPACE, BACKSPACE, PERIOD gestures
✓ Separate confidence thresholds for letters vs controls
✓ Real-time progress bars
✓ Professional dark UI design
✓ Detection history with color coding
✓ Audio-visual feedback for special gestures

REQUIRES:
    - alphabet_model.keras
    - alphabet_labels.json

CONTROLS:
    q         — Quit
    c         — Clear sentence
    SPACE key — Add space (backup)
    BKSP key  — Delete char (backup)

GESTURES:
    A-Z       → Alphabet letters
    SPACE     → Add space (open palm)
    BACKSPACE → Delete last char (swipe left)
    PERIOD    → Add period (closed fist)

HOW TO USE:
    python 4_alphabet_detector_ultimate.py
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
from collections import deque

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 30
MODEL_PATH = r"C:\MyWorks\projects\sign_language_alphabet\alphabet_model.keras"  # r"alphabet_model.keras"
LABEL_PATH = r"C:\MyWorks\projects\sign_language_alphabet\alphabet_labels.json"

# Detection settings
CONFIDENCE_THRESHOLD = 0.85                    # For alphabet letters
SPECIAL_CONFIDENCE_THRESHOLD = 0.92            # Higher for control gestures
STABLE_PREDICTIONS_REQUIRED = 10
MIN_HAND_FRAMES = 22

# Special gesture identifiers
SPECIAL_GESTURES = ["SPACE", "BACKSPACE", "PERIOD"]

# Window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Hand guide box
GUIDE_BOX_SIZE = 480
GUIDE_BOX_COLOR = (0, 220, 120)

# UI Colors
COLOR_LETTER = (0, 255, 150)         # Green for letters
COLOR_SPACE = (100, 200, 255)        # Blue for SPACE
COLOR_BACKSPACE = (255, 165, 0)      # Orange for BACKSPACE
COLOR_PERIOD = (255, 100, 255)       # Purple for PERIOD

# History
MAX_HISTORY = 6


# ─────────────────────────────────────────────
# LOAD MODEL AND LABELS
# ─────────────────────────────────────────────
def load_model_and_labels():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        print(f"[✗] Model not found: {MODEL_PATH}")
        print(f"    Run training first: python 3_train_alphabet_model.py")
        exit()

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)

    idx_to_label = {v: k for k, v in label_map.items()}

    print(f"[✓] Model: {os.path.abspath(MODEL_PATH)}")
    print(f"[✓] Signs: {len(idx_to_label)} total")
    print(f"    ├─ Alphabet: {len([k for k in idx_to_label.values() if k not in SPECIAL_GESTURES])}")
    print(f"    └─ Controls: {len([k for k in idx_to_label.values() if k in SPECIAL_GESTURES])}")

    return model, idx_to_label


# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

LANDMARK_STYLE = mp_draw.DrawingSpec(color=(0, 255, 150), thickness=3, circle_radius=4)
CONNECTION_STYLE = mp_draw.DrawingSpec(color=(50, 200, 255), thickness=3)


def extract_hand_landmarks(results):
    """Extracts landmarks from detected hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return [0.0] * 63 + landmarks
    return [0.0] * 126


def predict_sign(model, idx_to_label, sequence):
    """Predicts sign with gesture-specific confidence thresholds."""
    input_data = np.array([sequence])
    prediction = model.predict(input_data, verbose=0)[0]

    max_idx = np.argmax(prediction)
    confidence = float(prediction[max_idx])
    detected_label = idx_to_label[max_idx]
    
    # Apply appropriate threshold
    if detected_label in SPECIAL_GESTURES:
        threshold = SPECIAL_CONFIDENCE_THRESHOLD
    else:
        threshold = CONFIDENCE_THRESHOLD
    
    sign = detected_label if confidence >= threshold else None

    return sign, confidence


def get_sign_color(sign):
    """Returns color based on sign type."""
    if sign == "SPACE":
        return COLOR_SPACE
    elif sign == "BACKSPACE":
        return COLOR_BACKSPACE
    elif sign == "PERIOD":
        return COLOR_PERIOD
    else:
        return COLOR_LETTER


def draw_guide_box(frame, flash=False):
    """Draws animated guide box with optional flash effect."""
    h, w = frame.shape[:2]
    
    x1 = (w - GUIDE_BOX_SIZE) // 2
    y1 = (h - GUIDE_BOX_SIZE) // 2 - 70
    x2 = x1 + GUIDE_BOX_SIZE
    y2 = y1 + GUIDE_BOX_SIZE
    
    # Flash effect for special gestures
    box_color = (0, 255, 255) if flash else GUIDE_BOX_COLOR
    thickness = 5 if flash else 3
    
    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    
    # Corner markers
    corner_len = 45
    corner_thick = 6 if flash else 5
    
    corners = [
        ((x1, y1), (x1 + corner_len, y1), (x1, y1 + corner_len)),  # Top-left
        ((x2, y1), (x2 - corner_len, y1), (x2, y1 + corner_len)),  # Top-right
        ((x1, y2), (x1 + corner_len, y2), (x1, y2 - corner_len)),  # Bottom-left
        ((x2, y2), (x2 - corner_len, y2), (x2, y2 - corner_len)),  # Bottom-right
    ]
    
    for corner in corners:
        cv2.line(frame, corner[0], corner[1], box_color, corner_thick)
        cv2.line(frame, corner[0], corner[2], box_color, corner_thick)
    
    # Instruction
    cv2.putText(frame, "PLACE HAND HERE", (x1 + 130, y1 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, box_color, 2)


def draw_progress_bar(frame, x, y, width, height, progress, label, color):
    """Draws labeled progress bar."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), 2)
    
    # Fill
    fill_w = int(width * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
    
    # Label
    percent = int(progress * 100)
    label_text = f"{label}: {percent}%"
    cv2.putText(frame, label_text, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def draw_gesture_legend(frame):
    """Draws legend showing gesture meanings."""
    h, w = frame.shape[:2]
    
    # Legend panel
    legend_x = 20
    legend_y = h - 260
    legend_w = 320
    legend_h = 140
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h),
                  (80, 80, 80), 2)
    
    # Title
    cv2.putText(frame, "SPECIAL GESTURES:", (legend_x + 10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Gestures
    y_off = legend_y + 55
    gestures_info = [
        ("SPACE", "Open palm", COLOR_SPACE),
        ("BACKSPACE", "Swipe left", COLOR_BACKSPACE),
        ("PERIOD", "Closed fist", COLOR_PERIOD)
    ]
    
    for sign, desc, color in gestures_info:
        cv2.circle(frame, (legend_x + 20, y_off - 5), 6, color, -1)
        cv2.putText(frame, f"{sign}: {desc}", (legend_x + 35, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_off += 28


def draw_ui(frame, detected_sign, confidence, sentence, hand_detected, stable_count,
            hand_frames, detection_history, fps, flash_effect):
    """Draws complete professional UI."""
    h, w = frame.shape[:2]
    
    # ─── TOP PANEL ───
    cv2.rectangle(frame, (0, 0), (w, 100), (25, 25, 25), -1)
    cv2.line(frame, (0, 100), (w, 100), (0, 160, 160), 3)
    
    # Title
    cv2.putText(frame, "SIGN LANGUAGE ALPHABET DETECTOR", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 220, 220), 2)
    
    # Subtitle
    cv2.putText(frame, "with Smart Gesture Recognition", (22, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 180), 1)
    
    # Hand status
    status_color = (0, 255, 120) if hand_detected else (0, 100, 200)
    status_text = "HAND DETECTED" if hand_detected else "NO HAND"
    cv2.circle(frame, (20, 82), 9, status_color, -1)
    cv2.putText(frame, status_text, (40, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 150), 2)
    
    # ─── RIGHT PANEL ───
    panel_x = w - 360
    panel_y = 120
    panel_w = 340
    panel_h = 480
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (0, 160, 160), 2)
    
    # Section title
    cv2.putText(frame, "DETECTION STATUS", (panel_x + 12, panel_y + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 220), 2)
    
    # Current detection
    y_offset = panel_y + 75
    if detected_sign:
        sign_color = get_sign_color(detected_sign)
        
        cv2.putText(frame, "RECOGNIZED:", (panel_x + 12, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        
        # Sign type indicator
        sign_type = "[CONTROL]" if detected_sign in SPECIAL_GESTURES else "[LETTER]"
        cv2.putText(frame, sign_type, (panel_x + 12, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        cv2.putText(frame, detected_sign, (panel_x + 12, y_offset + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.9, sign_color, 3)
        
        cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%",
                    (panel_x + 12, y_offset + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)
    else:
        cv2.putText(frame, "Waiting for stable gesture...",
                    (panel_x + 12, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (110, 110, 110), 1)
    
    # Progress bars
    y_offset += 135
    
    # Hand stability
    hand_progress = min(hand_frames / MIN_HAND_FRAMES, 1.0)
    draw_progress_bar(frame, panel_x + 12, y_offset, 316, 28, hand_progress,
                     "Hand Visibility", (60, 160, 255))
    
    y_offset += 58
    
    # Sign stability
    stable_progress = stable_count / STABLE_PREDICTIONS_REQUIRED
    draw_progress_bar(frame, panel_x + 12, y_offset, 316, 28, stable_progress,
                     "Sign Stability", (0, 220, 120))
    
    # History
    y_offset += 78
    cv2.putText(frame, "RECENT HISTORY:", (panel_x + 12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 160, 160), 1)
    
    y_offset += 32
    if detection_history:
        for i, (hist_sign, hist_conf) in enumerate(detection_history):
            alpha = 1.0 - (i * 0.13)
            hist_color = get_sign_color(hist_sign)
            hist_color_faded = tuple(int(c * alpha) for c in hist_color)
            
            cv2.putText(frame, f"{hist_sign} ({hist_conf*100:.0f}%)",
                       (panel_x + 18, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.56, hist_color_faded, 1)
    else:
        cv2.putText(frame, "No detections yet", (panel_x + 18, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    
    # ─── BOTTOM PANEL ───
    bottom_h = 120
    cv2.rectangle(frame, (0, h - bottom_h), (w, h), (25, 25, 25), -1)
    cv2.line(frame, (0, h - bottom_h), (w, h - bottom_h), (0, 160, 160), 3)
    
    # Sentence label
    cv2.putText(frame, "OUTPUT:", (20, h - bottom_h + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (160, 160, 160), 1)
    
    # Sentence
    sentence_text = sentence if sentence else "Your message will appear here..."
    sentence_color = (255, 255, 255) if sentence else (90, 90, 90)
    
    max_chars = 68
    if len(sentence_text) > max_chars:
        line1 = sentence_text[:max_chars]
        line2 = sentence_text[max_chars:max_chars*2]
        cv2.putText(frame, line1, (20, h - bottom_h + 68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, sentence_color, 2)
        if line2:
            cv2.putText(frame, line2, (20, h - bottom_h + 98),
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, sentence_color, 2)
    else:
        cv2.putText(frame, sentence_text, (20, h - bottom_h + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, sentence_color, 2)
    
    # Controls
    cv2.putText(frame, "[Q] Quit  |  [C] Clear  |  [SPACE] Space  |  [BKSP] Delete",
                (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (110, 110, 110), 1)


def main():
    print("=" * 75)
    print("  SIGN LANGUAGE ALPHABET DETECTOR - ULTIMATE VERSION")
    print("=" * 75)
    print("\n  Loading model...\n")
    
    model, idx_to_label = load_model_and_labels()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Camera not accessible!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("\n  [✓] Camera ready")
    print("  [✓] Resolution: 1280x720")
    print("\n  ┌─ INSTRUCTIONS ─────────────────────────────────────┐")
    print("  │  1. Place hand in green guide box                  │")
    print("  │  2. Make clear alphabet sign or gesture            │")
    print("  │  3. Hold steady until both bars fill (100%)        │")
    print("  │  4. Character/action added automatically           │")
    print("  │  5. Remove hand before next sign                   │")
    print("  └─────────────────────────────────────────────────────┘")
    print("\n  ┌─ SPECIAL GESTURES ─────────────────────────────────┐")
    print("  │  SPACE     → Open palm facing camera               │")
    print("  │  BACKSPACE → Swipe hand left                        │")
    print("  │  PERIOD    → Closed fist                            │")
    print("  └─────────────────────────────────────────────────────┘")
    print("\n  ┌─ KEYBOARD SHORTCUTS ───────────────────────────────┐")
    print("  │  Q / ESC      → Quit application                    │")
    print("  │  C            → Clear sentence                      │")
    print("  │  SPACE        → Add space (backup)                  │")
    print("  │  BACKSPACE    → Delete char (backup)                │")
    print("  └─────────────────────────────────────────────────────┘")
    print("=" * 75 + "\n")
    
    # State
    sequence = []
    sentence = ""
    detection_history = deque(maxlen=MAX_HISTORY)
    prediction_buffer = deque(maxlen=STABLE_PREDICTIONS_REQUIRED)
    
    stable_count = 0
    last_confirmed_sign = None
    last_confirmation_time = 0
    flash_effect = False
    flash_start = 0
    
    prev_time = time.time()
    
    cv2.namedWindow("Alphabet Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Alphabet Detector", WINDOW_WIDTH, WINDOW_HEIGHT)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Flash effect timing
        if flash_effect and (curr_time - flash_start) > 0.3:
            flash_effect = False
        
        # Draw guide box
        draw_guide_box(frame, flash=flash_effect)
        
        # MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        
        hand_detected = results.multi_hand_landmarks is not None
        
        if hand_detected:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm,
                                      mp.solutions.hands.HAND_CONNECTIONS,
                                      LANDMARK_STYLE, CONNECTION_STYLE)
        
        # Collect landmarks
        landmarks = extract_hand_landmarks(results)
        sequence.append(landmarks)
        
        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]
        
        hand_frames = sum(1 for fd in sequence if sum(fd[63:]) > 0)
        
        # Prediction
        detected_sign = None
        confidence = 0
        
        if len(sequence) == SEQUENCE_LENGTH and hand_frames >= MIN_HAND_FRAMES:
            sign, confidence = predict_sign(model, idx_to_label, sequence)
            
            if sign:
                prediction_buffer.append(sign)
                
                if len(prediction_buffer) == STABLE_PREDICTIONS_REQUIRED:
                    if len(set(prediction_buffer)) == 1:
                        stable_sign = prediction_buffer[0]
                        time_since_last = curr_time - last_confirmation_time
                        
                        if stable_sign != last_confirmed_sign or time_since_last > 3.5:
                            detected_sign = stable_sign
                            
                            # Process gesture
                            if stable_sign == "SPACE":
                                sentence += " "
                                flash_effect = True
                                flash_start = curr_time
                            elif stable_sign == "BACKSPACE":
                                sentence = sentence[:-1]
                                flash_effect = True
                                flash_start = curr_time
                            elif stable_sign == "PERIOD":
                                sentence += "."
                                flash_effect = True
                                flash_start = curr_time
                            else:
                                sentence += stable_sign
                            
                            detection_history.appendleft((stable_sign, confidence))
                            last_confirmed_sign = stable_sign
                            last_confirmation_time = curr_time
                            prediction_buffer.clear()
                            
                            action = f"[{stable_sign}]" if stable_sign in SPECIAL_GESTURES else stable_sign
                            print(f"  [✓] {action} (conf: {confidence*100:.1f}%) → '{sentence}'")
                        else:
                            prediction_buffer.clear()
                    
                    stable_count = len([x for x in prediction_buffer if x == prediction_buffer[-1]])
                else:
                    stable_count = len([x for x in prediction_buffer if x == prediction_buffer[-1]]) if prediction_buffer else 0
            else:
                prediction_buffer.clear()
                stable_count = 0
        else:
            prediction_buffer.clear()
            stable_count = 0
        
        # Draw UI
        draw_ui(frame, detected_sign, confidence, sentence, hand_detected, stable_count,
                hand_frames, detection_history, fps, flash_effect)
        
        # Draw gesture legend
        draw_gesture_legend(frame)
        
        cv2.imshow("Alphabet Detector", frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            sentence = ""
            detection_history.clear()
            print("  [!] Sentence cleared")
        elif key == ord(' '):
            sentence += " "
            print(f"  [SPACE] → '{sentence}'")
        elif key == 8:
            sentence = sentence[:-1]
            print(f"  [BACKSPACE] → '{sentence}'")
    
    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    
    print("\n" + "=" * 75)
    print("  DETECTOR CLOSED")
    print("=" * 75)
    if sentence:
        print(f"\n  Final sentence: {sentence}\n")
    print("  Thank you for using the Sign Language Detector!\n")


if __name__ == "__main__":
    main()
