"""
ECG Signal Classification Using CNN + BiLSTM with Email Alert System
====================================================================
Classifies MIT-BIH Arrhythmia ECG signals into 5 classes and sends
an email alert based on the detection (Normal or Abnormal).

Compatible with macOS (tensorflow-macos / tensorflow-metal).
"""

import os
import sys
import ssl
import smtplib
import urllib.request
import zipfile
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "mitbih_train.csv")
TEST_FILE = os.path.join(DATA_DIR, "mitbih_test.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecg_cnn_bilstm.keras")

# Kaggle public dataset mirror links
TRAIN_URL = "https://huggingface.co/datasets/ayoubkirouane/MIT_BIH/resolve/main/mitbih_train.csv"
TEST_URL = "https://huggingface.co/datasets/ayoubkirouane/MIT_BIH/resolve/main/mitbih_test.csv"

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}

NUM_CLASSES = 5
EPOCHS = 20
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Data Functions
# ---------------------------------------------------------------------------
def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL with a progress indicator."""
    print(f"  Downloading: {os.path.basename(dest_path)}")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 64

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
            print()
        print(f"  ✓ Saved to {dest_path}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        raise


def load_data() -> tuple:
    """Load the MIT-BIH CSV files (downloads them if missing)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isfile(TRAIN_FILE):
        print("\n📥 Training data not found. Downloading...")
        download_file(TRAIN_URL, TRAIN_FILE)

    if not os.path.isfile(TEST_FILE):
        print("\n📥 Test data not found. Downloading...")
        download_file(TEST_URL, TEST_FILE)

    print("\n📂 Loading dataset...")
    train_df = pd.read_csv(TRAIN_FILE, header=None)
    test_df = pd.read_csv(TEST_FILE, header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values.astype(int)

    X_train = X_train.reshape(-1, X_train.shape[1], 1).astype(np.float32)
    X_test = X_test.reshape(-1, X_test.shape[1], 1).astype(np.float32)

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

    return X_train, y_train_cat, X_test, y_test_cat, y_test


def extract_signal_from_image(image_path: str) -> np.ndarray:
    """Read an ECG image, trace the signal, and resample to 187 points."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image file.")
    
    # Invert binary threshold to isolate dark line on lighter background
    # (Assuming traditional ECG printout: dark trace on white/grid paper)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    h, w = thresh.shape
    signal = []
    
    # Extract y-coordinate (height) for each x-coordinate (time)
    for x in range(w):
        col = thresh[:, x]
        y_indices = np.where(col > 0)[0]
        if len(y_indices) > 0:
            y_mean = np.mean(y_indices)
            signal.append(h - y_mean) # Invert y so signal peaks go upward
        else:
            if len(signal) > 0:
                signal.append(signal[-1])
            else:
                signal.append(h / 2)
                
    signal = np.array(signal)
    
    # Resample to exactly 187 points using linear interpolation
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, 187)
    f = interp1d(x_old, signal, kind='linear')
    resampled_signal = f(x_new)
    
    # Normalize to 0-1 range to match MIT-BIH dataset format
    min_val = np.min(resampled_signal)
    max_val = np.max(resampled_signal)
    if max_val - min_val > 0:
        resampled_signal = (resampled_signal - min_val) / (max_val - min_val)
    else:
        resampled_signal = np.zeros(187)
        
    return resampled_signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Model Functions
# ---------------------------------------------------------------------------
def build_model(input_shape: tuple) -> Model:
    """Build the CNN + BiLSTM model."""
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=5, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ECG_CNN_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_or_train_model() -> Model:
    """Load existing model or train a new one if it doesn't exist."""
    print("=" * 60)
    print("  ECG Signal Classification")
    print("=" * 60)

    if os.path.exists(MODEL_PATH):
        print(f"\n✅ Found existing model. Loading...")
        return load_model(MODEL_PATH)
    
    print("\n⚠️ No existing model found. Training a new model first...")
    X_train, y_train, X_test, y_test, y_test_raw = load_data()
    
    print("\n🏗️ Building model...")
    model = build_model(input_shape=(X_train.shape[1], 1))
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("\n🚀 Starting training...")
    model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
    )

    print("\n📊 Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Email Functions
# ---------------------------------------------------------------------------
def send_email_wrapper(recipient_email: str, subject: str, html_body: str) -> None:
    """Helper to send an HTML email using Gmail SMTP."""
    SENDER_EMAIL = os.environ.get("ECG_SENDER_EMAIL", "")
    APP_PASSWORD = os.environ.get("ECG_APP_PASSWORD", "")
    DISPLAY_NAME = "ECG Monitoring System"

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{DISPLAY_NAME} <{SENDER_EMAIL}>"
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    try:
        print(f"\n📧 Sending email report to {recipient_email}...")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        print("  ✓ Email sent successfully!")
    except Exception as e:
        print(f"  ✗ Failed to send email: {e}")


def send_normal_email(recipient_email: str) -> None:
    """Sends a visually rich HTML email indicating a normal heartbeat."""
    subject = "💚 Heart Health Update — Normal ECG Result"

    html_body = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#f4f7fa;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:30px auto;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);padding:36px 30px;text-align:center;">
      <div style="font-size:48px;">💚</div>
      <h1 style="color:#ffffff;margin:12px 0 4px;font-size:26px;font-weight:700;">Heart Health Update</h1>
      <p style="color:rgba(255,255,255,0.9);margin:0;font-size:15px;">Normal ECG Result</p>
    </td>
  </tr>
  <!-- Body -->
  <tr>
    <td style="padding:32px 30px;">
      <p style="color:#333;font-size:16px;line-height:1.7;margin:0 0 20px;">Hi there,</p>
      <p style="color:#333;font-size:16px;line-height:1.7;margin:0 0 24px;">Our ECG monitoring system has just finished analysing your recent heartbeat recording, and we have <strong>great news!</strong></p>

      <!-- Result Card -->
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#e8faf0;border-left:4px solid #2ecc71;border-radius:8px;margin-bottom:28px;">
        <tr>
          <td style="padding:20px 24px;">
            <p style="margin:0 0 6px;font-size:13px;color:#27ae60;font-weight:700;text-transform:uppercase;letter-spacing:1px;">What was found?</p>
            <p style="margin:0;font-size:16px;color:#2d3436;line-height:1.6;">The system detected a <strong>Normal healthy rhythm</strong>. No irregularities or abnormal beats were found in this sample.</p>
          </td>
        </tr>
      </table>

      <!-- Action Items -->
      <p style="font-size:14px;font-weight:700;color:#2d3436;margin:0 0 12px;">💡 What should you do?</p>
      <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">1. Keep up the good work! A healthy heart is a happy heart.</td></tr>
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">2. Continue tracking your health as usual.</td></tr>
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">3. If you ever feel unwell or experience chest pain, shortness of breath, or dizziness, always consult a doctor immediately.</td></tr>
      </table>
    </td>
  </tr>
  <!-- Footer -->
  <tr>
    <td style="background:#f8f9fa;padding:20px 30px;border-top:1px solid #eee;">
      <p style="margin:0;font-size:12px;color:#999;line-height:1.6;text-align:center;">⚠️ This is an automated alert generated by an AI-based ECG classification system. It is <strong>NOT</strong> a medical diagnosis. Always consult a qualified healthcare professional.</p>
      <p style="margin:8px 0 0;font-size:11px;color:#bbb;text-align:center;">ECG Monitoring System &bull; CNN + BiLSTM Deep Learning</p>
    </td>
  </tr>
</table>
</body>
</html>
"""
    send_email_wrapper(recipient_email, subject, html_body)


def send_abnormal_email(recipient_email: str, detection: dict) -> None:
    """Sends a visually rich HTML email alerting about an irregular heartbeat."""
    subject = "🚨 Heart Health Alert — Irregular Heartbeat Detected"

    condition_info = {
        "Supraventricular": (
            "a Supraventricular Premature Beat",
            "This means the heart's upper chambers (atria) produced an extra or "
            "early beat. It's one of the more common irregular rhythms and is "
            "often harmless, but frequent episodes should be evaluated by a doctor."
        ),
        "Ventricular": (
            "a Ventricular Premature Beat",
            "This means the heart's lower chambers (ventricles) produced an extra "
            "beat. Occasional ones are usually not dangerous, but if they happen "
            "often, it's important to get them checked by a healthcare professional."
        ),
        "Fusion": (
            "a Fusion Beat",
            "This is a heartbeat that looks like a mix of a normal beat and an "
            "abnormal one happening at the same time. It can sometimes indicate "
            "an underlying heart rhythm issue that should be reviewed by a doctor."
        ),
        "Unknown": (
            "an Unclassifiable Beat",
            "This heartbeat didn't fit into the standard categories, which could "
            "mean it's an unusual rhythm. A medical professional can help determine "
            "whether it's something to be concerned about."
        ),
    }

    label = detection["predicted_label"]
    condition_name, condition_desc = condition_info.get(
        label, ("an abnormal heartbeat", "Please consult a doctor for further evaluation.")
    )

    html_body = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background-color:#f4f7fa;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:30px auto;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#e74c3c 0%,#c0392b 100%);padding:36px 30px;text-align:center;">
      <div style="font-size:48px;">🚨</div>
      <h1 style="color:#ffffff;margin:12px 0 4px;font-size:26px;font-weight:700;">Heart Health Alert</h1>
      <p style="color:rgba(255,255,255,0.9);margin:0;font-size:15px;">Irregular Heartbeat Detected</p>
    </td>
  </tr>
  <!-- Body -->
  <tr>
    <td style="padding:32px 30px;">
      <p style="color:#333;font-size:16px;line-height:1.7;margin:0 0 20px;">Hi there,</p>
      <p style="color:#333;font-size:16px;line-height:1.7;margin:0 0 24px;">Our ECG monitoring system has just finished analysing your recent heartbeat recording and found something that <strong>needs your attention</strong>.</p>

      <!-- Alert Card -->
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#fdf0ed;border-left:4px solid #e74c3c;border-radius:8px;margin-bottom:24px;">
        <tr>
          <td style="padding:20px 24px;">
            <p style="margin:0 0 6px;font-size:13px;color:#c0392b;font-weight:700;text-transform:uppercase;letter-spacing:1px;">What was found?</p>
            <p style="margin:0;font-size:16px;color:#2d3436;line-height:1.6;">The system detected <strong>{condition_name}</strong> in the recording.</p>
          </td>
        </tr>
      </table>

      <!-- Explanation Card -->
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#fef9e7;border-left:4px solid #f39c12;border-radius:8px;margin-bottom:24px;">
        <tr>
          <td style="padding:20px 24px;">
            <p style="margin:0 0 6px;font-size:13px;color:#e67e22;font-weight:700;text-transform:uppercase;letter-spacing:1px;">What does this mean?</p>
            <p style="margin:0;font-size:15px;color:#2d3436;line-height:1.6;">{condition_desc}</p>
          </td>
        </tr>
      </table>

      <!-- Technical Details -->
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#eaf2f8;border-left:4px solid #2980b9;border-radius:8px;margin-bottom:24px;">
        <tr>
          <td style="padding:20px 24px;">
            <p style="margin:0 0 12px;font-size:13px;color:#2471a3;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Technical Details (for your doctor)</p>
            <table cellpadding="4" cellspacing="0" style="font-size:14px;color:#2d3436;">
              <tr><td style="font-weight:600;padding-right:12px;">Detected Type:</td><td>{detection['predicted_label']} (Class {detection['predicted_class']})</td></tr>
              <tr><td style="font-weight:600;padding-right:12px;">Actual Label:</td><td>{detection['true_label']}</td></tr>
            </table>
            <hr style="border:none;border-top:1px solid #d4e6f1;margin:14px 0;">
            <p style="margin:0 0 6px;font-size:12px;color:#7f8c8d;font-weight:600;">HEARTBEAT TYPES REFERENCE</p>
            <table cellpadding="2" cellspacing="0" style="font-size:13px;color:#555;">
              <tr><td style="font-weight:600;">Normal</td><td style="padding-left:10px;">Healthy, regular rhythm</td></tr>
              <tr><td style="font-weight:600;">Supraventricular</td><td style="padding-left:10px;">Extra beat from upper chambers</td></tr>
              <tr><td style="font-weight:600;">Ventricular</td><td style="padding-left:10px;">Extra beat from lower chambers</td></tr>
              <tr><td style="font-weight:600;">Fusion</td><td style="padding-left:10px;">Mixed normal + abnormal beat</td></tr>
              <tr><td style="font-weight:600;">Unknown</td><td style="padding-left:10px;">Unclassifiable rhythm</td></tr>
            </table>
          </td>
        </tr>
      </table>

      <!-- Action Items -->
      <p style="font-size:14px;font-weight:700;color:#2d3436;margin:0 0 12px;">💡 What should you do?</p>
      <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">1. Don't panic — one irregular beat is not always a cause for concern.</td></tr>
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">2. Save this email to share with your doctor at your next visit.</td></tr>
        <tr><td style="padding:6px 0 6px 16px;color:#555;font-size:15px;line-height:1.6;">3. If you experience symptoms like dizziness, chest pain, or shortness of breath, seek medical attention promptly.</td></tr>
      </table>
    </td>
  </tr>
  <!-- Footer -->
  <tr>
    <td style="background:#f8f9fa;padding:20px 30px;border-top:1px solid #eee;">
      <p style="margin:0;font-size:12px;color:#999;line-height:1.6;text-align:center;">⚠️ This is an automated alert generated by an AI-based ECG classification system. It is <strong>NOT</strong> a medical diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment.</p>
      <p style="margin:8px 0 0;font-size:11px;color:#bbb;text-align:center;">ECG Monitoring System &bull; CNN + BiLSTM Deep Learning</p>
    </td>
  </tr>
</table>
</body>
</html>
"""
    send_email_wrapper(recipient_email, subject, html_body)


# ---------------------------------------------------------------------------
# Interactive Menu
# ---------------------------------------------------------------------------
def interactive_mode(model: Model) -> None:
    """Prompt user for email, then let them pick or upload a sample."""
    print("=" * 60)
    print("  ECG Interactive Analysis Menu")
    print("=" * 60)

    recipient_email = input("\n📬 Enter your email for the results report: ").strip()
    if not recipient_email or "@" not in recipient_email:
        print("  ✗ Invalid email address. Exiting.")
        sys.exit(1)

    print("\nHow would you like to provide the ECG data?")
    print("  1. Pick a random NORMAL heartbeat")
    print("  2. Pick a random ABNORMAL (Irregular) heartbeat")
    print("  3. Upload my own CSV file (single row of 188 features)")
    print("  4. Upload an Image of an ECG graph (PNG, JPG)")
    choice = input("👉 Enter choice (1, 2, 3, or 4): ").strip()

    sample_x = None
    true_class = -1

    if choice in ["1", "2"]:
        print("\n📂 Loading dataset for random sample...")
        _, _, X_test, _, y_test_raw = load_data()
        
        if choice == "1":
            # Normal beat (class 0)
            indices = np.where(y_test_raw == 0)[0]
        else:
            # Abnormal beat (class > 0)
            indices = np.where(y_test_raw > 0)[0]
            
        idx = np.random.choice(indices)
        sample_x = X_test[idx:idx+1]
        true_class = int(y_test_raw[idx])
        print(f"  ✓ Selected sample #{idx} (True Class: {CLASS_NAMES[true_class]})")
    
    elif choice == "3":
        print("\n📁 Please select your CSV file from the window...")
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True) # Bring to front
        file_path = filedialog.askopenfilename(
            title="Select ECG CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            print("  ✗ No file selected.")
            sys.exit(1)
            
        try:
            df = pd.read_csv(file_path, header=None)
            data = df.iloc[0].values
            if len(data) >= 188:
                sample_x = data[:187].reshape(1, 187, 1).astype(np.float32)
                true_class = int(data[187])
            elif len(data) == 187:
                sample_x = data.reshape(1, 187, 1).astype(np.float32)
                true_class = -1
            else:
                print(f"  ✗ Expected 187 or 188 columns, got {len(data)}.")
                sys.exit(1)
            print("  ✓ Loaded custom sample successfully.")
        except Exception as e:
            print(f"  ✗ Failed to load CSV: {e}")
            sys.exit(1)
            
    elif choice == "4":
        print("\n🖼️ Please select your ECG image from the window...")
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True) # Bring to front
        file_path = filedialog.askopenfilename(
            title="Select ECG Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if not file_path:
            print("  ✗ No file selected.")
            sys.exit(1)
            
        try:
            print("\n⚙️  Processing image and extracting signal...")
            signal_data = extract_signal_from_image(file_path)
            sample_x = signal_data.reshape(1, 187, 1)
            true_class = -1
            print("  ✓ Processed image successfully.")
        except Exception as e:
            print(f"  ✗ Failed to process image: {e}")
            sys.exit(1)
            
    else:
        print("  ✗ Invalid choice.")
        sys.exit(1)

    print("\n🔍 Analyzing ECG signal...")
    preds = model.predict(sample_x, verbose=0)
    pred_class = int(np.argmax(preds, axis=1)[0])
    pred_label = CLASS_NAMES[pred_class]
    confidence = float(np.max(preds)) * 100

    print(f"\n  Result: {pred_label} (Confidence: {confidence:.2f}%)")

    if pred_class == 0:
        send_normal_email(recipient_email)
    else:
        detection = {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "true_class": true_class if true_class != -1 else "N/A",
            "true_label": CLASS_NAMES.get(true_class, "Unknown/Unprovided") if true_class != -1 else "N/A",
        }
        send_abnormal_email(recipient_email, detection)


def main() -> None:
    model = get_or_train_model()
    interactive_mode(model)


if __name__ == "__main__":
    main()
