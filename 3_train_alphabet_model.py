"""
TRAIN ALPHABET MODEL
=====================
Trains an LSTM model on:
    - A-Z alphabet gestures
    - SPACE, BACKSPACE, PERIOD control gestures

OUTPUT:
    - alphabet_model.keras
    - alphabet_labels.json
    - alphabet_training_report.txt

HOW TO USE:
    python 3_train_alphabet_model.py
"""

import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
SPECIAL_SIGNS = ["SPACE", "BACKSPACE", "PERIOD"]
ALL_SIGNS = ALPHABETS + SPECIAL_SIGNS

SEQUENCE_LENGTH = 30
DATA_DIR = r"C:\MyWorks\projects\sign_language_alphabet\alphabet_dataset"  # "alphabet_dataset"
MODEL_PATH = r"C:\\MyWorks\\projects\\sign_language_alphabet\\alphabet_model.keras"
LABEL_PATH = r"C:\\MyWorks\\projects\\sign_language_alphabet\\alphabet_labels.json"
REPORT_PATH = r"C:\\MyWorks\\projects\\sign_language_alphabet\\alphabet_training_report.txt"  # "alphabet_training_report.txt"

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001


def load_dataset():
    """
    Loads all .npy files from alphabet_dataset/.
    Returns X (samples, 30, 126) and y (samples,) integer labels.
    """
    X, y = [], []

    for label_idx, sign in enumerate(ALL_SIGNS):
        sign_path = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_path):
            print(f"  [!] Skipping '{sign}' — no data folder found")
            continue

        files = sorted([f for f in os.listdir(sign_path) if f.endswith(".npy")])
        if len(files) == 0:
            print(f"  [!] Skipping '{sign}' — folder is empty")
            continue

        for file in files:
            filepath = os.path.join(sign_path, file)
            sequence = np.load(filepath)
            X.append(sequence)
            y.append(label_idx)

        print(f"  [✓] Loaded {len(files):>3} samples for '{sign}'")

    X = np.array(X)
    y = np.array(y)
    return X, y


def build_model(input_shape, num_classes):
    """
    LSTM model for alphabet recognition.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates and generates classification report."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(model.predict(X_test), axis=1)

    report = classification_report(y_test, y_pred, target_names=ALL_SIGNS)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return report, conf_matrix


def main():
    print("=" * 60)
    print("  ALPHABET MODEL — TRAINING")
    print("=" * 60)

    # Load data
    print("\n[1] Loading dataset...\n")
    X, y = load_dataset()

    if len(X) == 0:
        print("\n[✗] No data found! Run the preparation scripts first.")
        return

    print(f"\n  Total samples loaded : {len(X)}")
    print(f"  Sequence shape       : {X.shape}")
    print(f"  Number of signs      : {len(np.unique(y))}")

    # Split data
    print("\n[2] Splitting data into train/test sets...\n")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training samples     : {len(X_train)}")
    print(f"  Testing samples      : {len(X_test)}")

    # Build model
    print("\n[3] Building LSTM model...\n")
    import tensorflow as tf

    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=len(ALL_SIGNS)
    )
    model.summary()

    # Train
    print("\n[4] Training model...\n")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )

    # Evaluate
    print("\n[5] Evaluating model...\n")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")

    report, conf_matrix = evaluate_model(model, X_test, y_test)
    print("\n  Classification Report:\n")
    print(report)

    # Save
    print("[6] Saving model and labels...\n")

    model.save(MODEL_PATH)
    print(f"  [✓] Model saved to '{MODEL_PATH}'")

    label_map = {sign: idx for idx, sign in enumerate(ALL_SIGNS)}
    with open(LABEL_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  [✓] Labels saved to '{LABEL_PATH}'")

    with open(REPORT_PATH, "w") as f:
        f.write("ALPHABET MODEL — TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Samples     : {len(X)}\n")
        f.write(f"Training Samples  : {len(X_train)}\n")
        f.write(f"Testing Samples   : {len(X_test)}\n")
        f.write(f"Epochs            : {EPOCHS}\n")
        f.write(f"Test Accuracy     : {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss         : {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write("-" * 60 + "\n")
        f.write(str(conf_matrix))
    print(f"  [✓] Report saved to '{REPORT_PATH}'")

    print("\n" + "=" * 60)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Accuracy: {test_acc * 100:.2f}%")
    print("  Next step: Run  python 4_alphabet_detector.py\n")


if __name__ == "__main__":
    main()
