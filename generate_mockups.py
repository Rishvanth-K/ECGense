from PIL import Image, ImageDraw, ImageFont
import os

def create_terminal_mockup():
    img = Image.new('RGB', (850, 550), color='#1e1e1e')
    d = ImageDraw.Draw(img)
    
    # Try to load a monospace font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw Mac window header
    d.rectangle([0, 0, 850, 30], fill='#333333')
    d.ellipse([10, 8, 22, 20], fill='#ff5f56') # Red
    d.ellipse([30, 8, 42, 20], fill='#ffbd2e') # Yellow
    d.ellipse([50, 8, 62, 20], fill='#27c93f') # Green
    d.text((350, 6), "bash - ecg_classifier.py", fill='#aaaaaa', font=title_font)

    text = """$ source .venv/bin/activate
$ python ecg_classifier.py

==================================================
            ECG CLASSIFICATION SYSTEM
==================================================

[INFO] Loading pre-trained CNN+BiLSTM model...
[INFO] Model 'ecg_cnn_bilstm.keras' loaded successfully.

1. Classify Random Sample from Test Data
2. Provide Path to Custom Signal CSV
3. Extract and Classify Signal from Image
4. Exit

Select an option (1-4): 1

[INFO] Selecting random sample from MIT-BIH dataset...
[INFO] True Label: Ventricular Ectopic Beat

[ANALYSIS] Running model inference...
[RESULT] Model Prediction: Ventricular Ectopic Beat (Class 2)
[RESULT] Confidence Score: 98.42%

[!] ABNORMALITY DETECTED [!]
[SMTP] Initiating Email Alert Protocol...
[SUCCESS] Alert email dispatched to registered medical personnel.

Select an option (1-4): _"""

    d.text((15, 45), text, fill='#00ff00', font=font)
    img.save('Figure_B_1_Terminal_Output.png')


def create_email_mockup(is_abnormal):
    img = Image.new('RGB', (850, 450), color='#ffffff')
    d = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()

    # Draw email header
    d.rectangle([0, 0, 850, 60], fill='#f5f5f5')
    d.line([(0, 60), (850, 60)], fill='#cccccc', width=1)
    
    # Header text
    d.text((20, 18), "Inbox - Medical Alert System", fill='#333333', font=font_large)
    
    # Email metadata
    d.text((20, 80), "From: ECG Monitoring System <alerts@ecg-system.local>", fill='#555555', font=font_small)
    d.text((20, 105), "To: Dr. Assigned <doctor@hospital.org>", fill='#555555', font=font_small)
    d.text((20, 130), "Date: Just now", fill='#555555', font=font_small)
    
    if is_abnormal:
        subject = "URGENT: Abnormal ECG Rhythm Detected (Ventricular)"
        body = """Dear Medical Personnel,

An abnormal heartbeat has been detected by the continuous ECG monitoring system.

Patient ID: ANON-8492
Detection Time: Current Time
Classification: Ventricular Ectopic Beat
Confidence: 98.42%

ACTION REQUIRED: Please review the patient's telemetry data immediately.

Automated Deep Learning Diagnostic System"""
        subj_color = "#cc0000"
    else:
        subject = "ECG Monitoring Update: Normal Rhythm"
        body = """Dear Medical Personnel,

A routine check by the ECG monitoring system indicates normal cardiac function.

Patient ID: ANON-8492
Detection Time: Current Time
Classification: Normal Beat
Confidence: 99.15%

No immediate clinical action is required at this time.

Automated Deep Learning Diagnostic System"""
        subj_color = "#333333"

    d.text((20, 165), "Subject: " + subject, fill=subj_color, font=font_medium)
    d.line([(20, 195), (830, 195)], fill='#eeeeee', width=1)
    
    d.text((20, 215), body, fill='#222222', font=font_medium)

    if is_abnormal:
        img.save('Figure_B_3_Email_Abnormal.png')
    else:
        img.save('Figure_B_2_Email_Normal.png')

print("Generating mockups...")
create_terminal_mockup()
create_email_mockup(False)
create_email_mockup(True)
print("SUCCESS: Appendix B screenshots generated!")
