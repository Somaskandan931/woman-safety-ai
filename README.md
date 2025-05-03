# 🛡️ AI-Powered Woman Safety System Using Computer Vision & Voice Detection

An intelligent AI-powered surveillance and emergency alert system that enhances women's safety by using real-time computer vision, pose recognition, and voice keyword detection. When suspicious activities (like falling, raised hands) or emergency keywords (like "help", "save me") are detected, the system immediately alerts guardians or authorities via SMS/WhatsApp with location and image evidence.

---

## 📑 Table of Contents

- 🚀 Features
- 💡 Tech Stack
- 🧠 Architecture
- 🛠️ Installation
- ▶️ Usage
- 📳 Alert System
- 🧠 Model Training
- 🔮 Future Enhancements
- ⚠️ Disclaimer
- 📜 License

---

## 🚀 Features

- ✅ Real-time object & person detection using YOLOv8
- ✅ Pose recognition (e.g., falling, hands raised) with MediaPipe
- ✅ Voice-based emergency detection (e.g., "help", "save me")
- ✅ Auto-alerts via SMS/WhatsApp with image & GPS location
- ✅ Evidence storage (image, optional audio)
- ✅ Streamlit-based dashboard for live monitoring (optional)
- ✅ Edge-device deployable (Raspberry Pi, Jetson Nano, etc.)

---

## 💡 Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python |
| Vision Models | YOLOv8 (Ultralytics), MediaPipe Pose |
| Audio Input | `SpeechRecognition`, `PyAudio` / `Vosk` |
| Alerting | Twilio API / WhatsApp Business API |
| Backend | FastAPI / Flask |
| Dashboard | Streamlit (optional) |
| Storage (opt.) | Firebase / MongoDB |
| Deployment | Render / Heroku / Raspberry Pi |

---

## 🧠 Architecture

```
[Live Camera Feed]
       ↓
[YOLOv8 + MediaPipe Pose Estimation]     [Microphone Input]
       ↓                                        ↓
[Threat/Pose Detection Logic]         [Voice Keyword Detection]
       ↓                                        ↓
       └──────> [Trigger Alert System] <───────┘
                         ↓
         [Send SMS/WhatsApp + Save Image + Location]

```

---

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/woman-safety-ai.git
cd woman-safety-ai
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup YOLOv8

```bash
pip install ultralytics
```

Or download YOLOv8n weights from [Ultralytics](https://github.com/ultralytics/ultralytics).

### 5. Setup Twilio for SMS/WhatsApp

Create a `.env` file:

```
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE=+1415XXXXXXX
USER_PHONE=+91XXXXXXXXXX
```

---

## ▶️ Usage

### Run the Main App:

```bash
python main.py
```

### What Happens:

- Live video feed starts.
- Detects people, pose (falling, hands-up), and voice keywords.
- If danger detected:
    - Saves a snapshot.
    - Sends emergency alert via SMS/WhatsApp.
    - Optionally logs the event to DB.

### Optional: Run Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

## 📳 Alert System

### Trigger Conditions:

- Detected fall or abnormal pose.
- Raised hands or sudden crouch.
- Recognized emergency voice command.

### Alert Actions:

- Save image frame.
- Capture GPS coordinates (optional).
- Send:
    - 📩 SMS or WhatsApp message.
    - 📷 Attached frame image.
    - 📍 Location URL (if enabled).

### Alert Tech:

- Twilio API (SMS/WhatsApp)
- Dotenv for secret management
- Firebase/MongoDB (optional for logging)

---

## 🧠 Model Training

### 🔍 YOLOv8 – Real-Time Object Detection

- Pretrained weights from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Can be retrained on custom CCTV or threat datasets.

### 🕺 Pose Estimation – MediaPipe

- Tracks human skeleton points.
- Logic for:
    - Fall detection.
    - Raised arms.
    - Abnormal crouching or lying posture.

### 🗣️ Voice Recognition – Emergency Detection

- Keywords: “Help”, “Save me”, “Please stop”.
- Uses `speech_recognition` with Google API (or Vosk for offline).
- Triggers alert same as visual threat detection.

---

## 🔮 Future Enhancements

- 🧠 Deep learning–based behavior classification
- 👤 Facial recognition to flag known attackers
- 📡 Real-time GPS integration via mobile
- 📱 Android app for alert receiver
- 🌐 Web UI or Progressive Web App (PWA)
- 🧲 IoT wearable emergency button
- ☁️ Cloud syncing & dashboard

---

## ⚠️ Disclaimer

This system is a **prototype** intended for research, demonstration, and learning purposes. It **should not replace** any legal, medical, or emergency systems. Always consult proper authorities in real emergencies.

---

## 📜 License

MIT License © 2025 [Your Name]

---

```bash
woman-safety-ai/
│
├── [main.py](http://main.py/)                        # Entry point: Captures feed, runs detection pipeline
├── [alert.py](http://alert.py/)                       # Sends SMS/WhatsApp alerts via Twilio
├── pose_detector.py              # Handles pose detection (MediaPipe/OpenPose logic)
├── object_detector.py           # YOLOv8 person and object detection logic
├── voice_detector.py            # (Optional) Voice keyword detection module
├── [dashboard.py](http://dashboard.py/)                  # Streamlit dashboard for real-time monitoring
│
├── utils/
│   ├── [helpers.py](http://helpers.py/)                # Utility functions (e.g., bounding box drawing, thresholding)
│   └── gps_utils.py              # Optional GPS fetching module
│
├── config/
│   ├── config.yaml               # Configurations like model thresholds, Twilio numbers, etc.
│   └── .env                      # Twilio and sensitive credentials (not tracked by Git)
│
├── data/
│   ├── snapshots/                # Stores snapshots/images when threats are detected
│   └── datasets/                 # (Optional) Datasets for training/fine-tuning
│
├── models/
│   ├── [yolov8n.pt](http://yolov8n.pt/)                # YOLOv8 weights
│   └── custom_pose_model.pth     # Custom pose model (if trained)
│
├── requirements.txt              # List of Python dependencies
├── [README.md](http://readme.md/)                     # Project overview and setup instructions
└── LICENSE                       # License information (MIT, Apache, etc.)
```