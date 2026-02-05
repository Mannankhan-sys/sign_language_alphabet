"""
ALPHABET DATASET PREPARATION
==============================
Extracts hand landmarks from your Kaggle alphabet images (A-Z)
and prepares them for LSTM training.

DATASET PATH:
    C:\MyWorks\projects\dataset\Sign Language for Alphabets\

STRUCTURE:
    A\a_1.jpg, a_2.jpg, ...
    B\b_1.jpg, b_2.jpg, ...
    ...
    Z\z_1.jpg, z_2.jpg, ...

OUTPUT:
    alphabet_dataset\
        A\0.npy, 1.npy, ...
        B\0.npy, 1.npy, ...
        ...
        SPACE\0.npy, ...  (you'll record these)
        BACKSPACE\0.npy, ...  (you'll record these)
        PERIOD\0.npy, ...  (you'll record these)

HOW TO USE:
    python 1_prepare_alphabet_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SOURCE_DIR = r"CC:\MyWorks\projects\dataset\alpha_2\ASL_Gestures_36_Classes\train"  # Change this to your dataset path
OUTPUT_DIR = r"C:\MyWorks\projects\sign_language_alphabet\alphabet_dataset" # Output folder for processed data
SEQUENCE_LENGTH = 30  # Each sample = 30 duplicate frames (since images are static)

ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

# Special signs you'll record later
SPECIAL_SIGNS = ["SPACE", "BACKSPACE", "PERIOD"]

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,  # We're processing static images
    max_num_hands=1,  # Alphabet signs use one hand
    min_detection_confidence=0.3
)


def extract_hand_landmarks(image_path):
    """
    Extracts hand landmarks from a single image.
    Returns 63 values [x0,y0,z0, ... x20,y20,z20] or None if no hand found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Take first hand
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return None


def create_sequence_from_image(landmarks):
    """
    Since we have static images but need sequences of 30 frames,
    we duplicate the same landmark data 30 times.
    
    Returns shape: (30, 126) where 126 = left_hand(63) + right_hand(63)
    For alphabet signs, we assume one hand, so we pad with zeros for the other.
    """
    sequence = []
    for _ in range(SEQUENCE_LENGTH):
        # right_hand = landmarks, left_hand = zeros
        frame_data = [0.0] * 63 + landmarks  # left(empty) + right(hand)
        sequence.append(frame_data)
    return np.array(sequence)


def process_alphabet_images():
    """
    Processes all alphabet images from A-Z.
    Extracts landmarks and saves as .npy sequences.
    """
    print("=" * 60)
    print("  PROCESSING ALPHABET IMAGES")
    print("=" * 60)

    total_processed = 0
    failed_images = []

    for letter in ALPHABETS:
        letter_dir = Path(SOURCE_DIR) / letter
        output_letter_dir = Path(OUTPUT_DIR) / letter
        output_letter_dir.mkdir(parents=True, exist_ok=True)

        if not letter_dir.exists():
            print(f"  [!] Folder not found: {letter}")
            continue

        # Get all image files
        image_files = list(letter_dir.glob("*.jpg")) + \
                      list(letter_dir.glob("*.jpeg")) + \
                      list(letter_dir.glob("*.png"))

        if len(image_files) == 0:
            print(f"  [!] No images found in: {letter}")
            continue

        print(f"\n  Processing '{letter}' — {len(image_files)} images found")

        success_count = 0
        for idx, img_path in enumerate(image_files):
            # Extract landmarks
            landmarks = extract_hand_landmarks(img_path)

            if landmarks is None:
                failed_images.append(str(img_path))
                continue

            # Create sequence (duplicate landmarks 30 times)
            sequence = create_sequence_from_image(landmarks)

            # Save
            output_path = output_letter_dir / f"{success_count}.npy"
            np.save(output_path, sequence)

            success_count += 1
            total_processed += 1

            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"    [{idx + 1}/{len(image_files)}] processed...")

        print(f"    [✓] {success_count} samples saved for '{letter}'")

    return total_processed, failed_images


def create_placeholder_for_special_signs():
    """
    Creates empty folders for SPACE, BACKSPACE, PERIOD.
    You'll record these gestures later using a separate script.
    """
    print(f"\n  Creating folders for special signs...")
    for sign in SPECIAL_SIGNS:
        sign_dir = Path(OUTPUT_DIR) / sign
        sign_dir.mkdir(parents=True, exist_ok=True)
        print(f"    [✓] {sign} folder created (record these later)")


def main():
    print("\n  Source: " + SOURCE_DIR)
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}\n")

    # Process alphabet images
    total, failed = process_alphabet_images()

    # Create folders for special signs
    create_placeholder_for_special_signs()

    # Summary
    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Total samples processed: {total}")
    print(f"  Failed images: {len(failed)}")

    if len(failed) > 0 and len(failed) <= 10:
        print(f"\n  Failed image paths:")
        for path in failed[:10]:
            print(f"    {path}")

    print(f"\n  Dataset structure:")
    for letter in ALPHABETS:
        letter_dir = Path(OUTPUT_DIR) / letter
        if letter_dir.exists():
            count = len(list(letter_dir.glob("*.npy")))
            print(f"    {letter}: {count} samples")

    print(f"\n  Special signs (record these next):")
    for sign in SPECIAL_SIGNS:
        print(f"    {sign}: 0 samples (use record_special_signs.py)")

    print(f"\n  Next steps:")
    print(f"    1. Run: python 2_record_special_signs.py")
    print(f"       (Record SPACE, BACKSPACE, PERIOD gestures)")
    print(f"    2. Run: python 3_train_alphabet_model.py")
    print(f"    3. Run: python 4_alphabet_detector.py\n")

    mp_hands.close()


if __name__ == "__main__":
    main()
