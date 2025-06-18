Real-Time Adaptive Workstation Monitoring System
This project implements a real-time intelligent monitoring system that utilizes webcam input to assess a user's emotional state, gaze direction, blinking behavior, and cognitive load. Based on these assessments, the system adaptively modifies the user's digital environment—such as minimizing distractions, playing calming music, or adjusting screen brightness—to support productivity, safety, and mental well-being.

Project Overview
The system uses deep learning models deployed via ONNX Runtime to process live webcam feeds. It integrates gaze detection, emotion classification, blink rate analysis, and cognitive load estimation into a unified monitoring application. Outputs are provided in real-time through a JSON file and a dynamic visual dashboard.

Core Functionalities
Emotion Detection: Classifies user emotion into categories such as happiness, sadness, anger, fear, etc.

Gaze Estimation: Detects whether the user is looking at the screen or away.

Blink Detection: Calculates blink frequency per minute to infer alertness or fatigue.

Cognitive Load Estimation: Uses a combination of blink rate and emotion to infer cognitive load levels (high, medium, low).

State-based Responses: Triggers automatic actions like locking the workstation, reducing brightness, or playing audio based on prolonged emotional states or inattentiveness.

Environment Adaptation Examples
User looking away for too long → All windows are minimized.

Sustained sadness detected → Calming music is played.

High cognitive load detected → Screen brightness is lowered.

No face detected for a minute → System locks automatically.

Dataset References
This system uses pre-trained models that can be trained or fine-tuned using publicly available datasets, such as:

FER2013 / FER+ for facial emotion recognition

MPIIGaze for gaze direction detection

Open/Closed Eye datasets for blink detection (optional)

RAVDESS for future extensions with voice-based emotion detection (optional)

Datasets can be downloaded from platforms such as Kaggle or respective official sources.

How to Use
Install the required Python libraries:

OpenCV

MediaPipe

ONNX Runtime

NumPy

Pygame

Screen Brightness Control

PyGetWindow

Place the following ONNX model files in the working directory:

model.onnx – Emotion recognition model

balanced_gaze.onnx – Gaze estimation model

(Optional) Eye state ONNX model for blink classification

Run the Python script to start the monitoring system.

The output will be shown on-screen and logged in realtime_log.json.

Applications
Mental wellness support during remote work

Intelligent digital assistant in workspaces

Personalized productivity and focus management

Research in human-computer interaction (HCI)

Credits
Developed as a real-time AI-powered interface for adaptive state detection using computer vision and deep learning models. Contributions include model integration, emotion/gaze logic, cognitive state estimation, and real-time system actions
