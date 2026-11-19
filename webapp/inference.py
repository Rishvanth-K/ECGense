"""ECGense inference: loads the existing trained CNN-BiLSTM model and classifies beats.

The class names and the image-to-signal logic are taken from ecg_classifier.py.
They are copied here (not imported) because ecg_classifier.py also imports tkinter
and the email code, which a web server inside a container does not need.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from tensorflow.keras.models import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT, "ecg_cnn_bilstm.keras"))
SAMPLES_PATH = os.path.join(ROOT, "samples", "sample_beats.csv")

SIGNAL_LENGTH = 187
MAX_BEATS = 50

# Same labels as CLASS_NAMES in ecg_classifier.py
CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}

_model = None


def get_model():
    """Load the trained model once and reuse it for every request."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def classify(signals):
    """signals: list of beats, each 187 values in 0-1. Returns one result per beat."""
    x = np.asarray(signals, dtype=np.float32).reshape(-1, SIGNAL_LENGTH, 1)
    probs = get_model().predict(x, verbose=0)
    results = []
    for beat, p in zip(x, probs):
        cls = int(np.argmax(p))
        results.append(
            {
                "predicted_class": cls,
                "predicted_label": CLASS_NAMES[cls],
                "confidence": round(float(p[cls]) * 100, 2),
                "probabilities": {CLASS_NAMES[i]: round(float(v) * 100, 2) for i, v in enumerate(p)},
                "signal": [round(float(v), 4) for v in beat.ravel()],
            }
        )
    return results


def parse_csv(text):
    """Same rules as option 3 in ecg_classifier.py: 188+ columns -> first 187 values
    plus the true label in column 188; exactly 187 columns -> no label.
    Unlike the CLI (first row only), every row is read, up to MAX_BEATS."""
    signals, labels = [], []
    for n, line in enumerate(text.strip().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = [float(v) for v in line.split(",")]
        except ValueError:
            raise ValueError(f"Row {n} contains non-numeric values")
        if len(data) >= SIGNAL_LENGTH + 1:
            signals.append(data[:SIGNAL_LENGTH])
            labels.append(int(data[SIGNAL_LENGTH]))
        elif len(data) == SIGNAL_LENGTH:
            signals.append(data)
            labels.append(None)
        else:
            raise ValueError(f"Row {n}: expected 187 or 188 columns, got {len(data)}")
    if not signals:
        raise ValueError("The CSV file is empty")
    if len(signals) > MAX_BEATS:
        raise ValueError(f"Too many rows; upload at most {MAX_BEATS} beats")
    return signals, labels


def extract_signal_from_image(image_bytes):
    """Read an ECG image, trace the signal, and resample to 187 points.
    Same algorithm as extract_signal_from_image() in ecg_classifier.py, but it reads
    the image from uploaded bytes and uses np.interp (linear, like scipy's interp1d)."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image file.")

    # Dark trace on light background -> inverted binary image
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    signal = []
    for x in range(w):
        y_indices = np.where(thresh[:, x] > 0)[0]
        if len(y_indices) > 0:
            signal.append(h - np.mean(y_indices))  # invert y so peaks go upward
        else:
            signal.append(signal[-1] if signal else h / 2)
    signal = np.array(signal)

    resampled = np.interp(np.linspace(0, 1, SIGNAL_LENGTH), np.linspace(0, 1, len(signal)), signal)

    lo, hi = resampled.min(), resampled.max()
    if hi - lo > 0:
        resampled = (resampled - lo) / (hi - lo)
    else:
        resampled = np.zeros(SIGNAL_LENGTH)
    return resampled.astype(np.float32).tolist()


def load_samples():
    """Real beats from mitbih_test.csv (4 per class), shown as demo buttons."""
    with open(SAMPLES_PATH) as f:
        signals, labels = parse_csv(f.read())
    return [
        {"id": i, "true_class": lab, "true_label": CLASS_NAMES[lab], "signal": sig}
        for i, (sig, lab) in enumerate(zip(signals, labels))
    ]
