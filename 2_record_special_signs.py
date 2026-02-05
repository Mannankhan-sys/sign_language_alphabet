"""
RECORD SPECIAL SIGNS
=====================
Record gestures for SPACE, BACKSPACE, and PERIOD.

These are control gestures that help you:
    SPACE     — Add a space between words
    BACKSPACE — Delete the last character
    PERIOD    — End a sentence

Make them DISTINCT from your alphabet signs!

SUGGESTIONS:
    SPACE     → Open palm facing camera
    BACKSPACE → Swipe hand left
    PERIOD    → Fist or closed hand

HOW TO USE:
    python 2_record_special_signs.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SPECIAL_SIGNS = ["SPACE", "BACKSPACE", "PERIOD"]
SAMPLES_PER_SIGN = 30
SEQUENCE_LENGTH = 30
OUTPUT_DIR = r"C:\MyWorks\projects\sign_language_alphabet\alphabet_dataset"

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


def extract_hand_landmarks(results):
    """Extracts landmarks from one hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return [0.0] * 63 + landmarks  # left_hand(empty) + right_hand
    return [0.0] * 126  # Both hands empty


def count_existing(sign):
    """Count existing samples."""
    path = os.path.join(OUTPUT_DIR, sign)
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(".npy")])


def record_sign(sign, suggestion, cap):
    """Record samples for one special sign."""
    os.makedirs(os.path.join(OUTPUT_DIR, sign), exist_ok=True)
    existing = count_existing(sign)

    if existing >= SAMPLES_PER_SIGN:
        print(f"  [DONE] '{sign}' already has {existing} samples.")
        return

    remaining = SAMPLES_PER_SIGN - existing

    print(f"\n{'=' * 60}")
    print(f"  RECORDING: {sign}")
    print(f"{'=' * 60}")
    print(f"  Suggestion: {suggestion}")
    print(f"  Already recorded: {existing} | Need: {remaining}")
    print(f"\n  Press SPACE to start recording...")
    print(f"{'=' * 60}\n")

    # Wait for space bar
    while True:
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Sign: {sign}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, suggestion, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{existing}/{SAMPLES_PER_SIGN} samples", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
        cv2.putText(frame, "Press SPACE to start", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow("Record Special Signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('q'):
            return

    # Record samples
    for sample_idx in range(existing, SAMPLES_PER_SIGN):
        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"Get Ready... {countdown}", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)
            cv2.putText(frame, f"{sign}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.imshow("Record Special Signs", frame)
            cv2.waitKey(1000)

        # Capture 30 frames
        sequence = []
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4)
                    )

            # Extract landmarks
            landmarks = extract_hand_landmarks(results)
            sequence.append(landmarks)

            # Progress
            progress = int((frame_idx / SEQUENCE_LENGTH) * 100)
            bar_width = int((frame_idx / SEQUENCE_LENGTH) * 500)
            cv2.rectangle(frame, (30, 400), (530, 450), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, 400), (30 + bar_width, 450), (0, 255, 100), -1)
            cv2.putText(frame, f"Recording... {progress}%", (200, 435),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Sample {sample_idx + 1}/{SAMPLES_PER_SIGN}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, sign, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Record Special Signs", frame)
            cv2.waitKey(1)

        # Save
        filepath = os.path.join(OUTPUT_DIR, sign, f"{sample_idx}.npy")
        np.save(filepath, np.array(sequence))

        time.sleep(0.5)

    print(f"  [✓] Recorded {SAMPLES_PER_SIGN} samples for '{sign}'")


def main():
    print("=" * 60)
    print("  RECORD SPECIAL CONTROL SIGNS")
    print("=" * 60)
    print("\n  You'll record 3 special gestures:")
    print("    SPACE     — Adds a space between words")
    print("    BACKSPACE — Deletes last character")
    print("    PERIOD    — Ends a sentence")
    print("\n  Make them VERY different from alphabet signs!")
    print("=" * 60)

    input("\n  Press ENTER to begin...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Camera not found!")
        return

    # Record each special sign
    record_sign("SPACE", "Open palm facing camera", cap)
    record_sign("BACKSPACE", "Swipe hand left", cap)
    record_sign("PERIOD", "Closed fist", cap)

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()

    # Summary
    print("\n" + "=" * 60)
    print("  RECORDING COMPLETE")
    print("=" * 60 + "\n")

    for sign in SPECIAL_SIGNS:
        count = count_existing(sign)
        print(f"    {sign:>10} : {count}/{SAMPLES_PER_SIGN} samples")

    print(f"\n  Next step: python 3_train_alphabet_model.py\n")


if __name__ == "__main__":
    main()
