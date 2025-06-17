import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import time
import json
import warnings
import os
import ctypes
import threading
import screen_brightness_control as sbc
from pygetwindow import getWindowsWithTitle
from playsound import playsound
import pygame

warnings.filterwarnings("ignore")

# === Constants ===
GAZE_MODEL = "balanced_gaze.onnx"
FACE_MODEL = "model.onnx"
LABELS_GAZE = ["down", "left", "right", "straight", "up"]
LABELS_EMOTION = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
NEGATIVE_EMOTIONS = {"anger", "contempt", "disgust", "fear", "sadness"}
IMG_GAZE_SIZE = (64, 64)
IMG_FACE_SIZE = (224, 224)
CLOSED_THRESHOLD = 5.0

# === Load models ===
sess_gaze = ort.InferenceSession(GAZE_MODEL)
sess_face = ort.InferenceSession(FACE_MODEL)
input_gaze = sess_gaze.get_inputs()[0].name
output_gaze = sess_gaze.get_outputs()[0].name
input_face = sess_face.get_inputs()[0].name
output_face = sess_face.get_outputs()[0].name

# === Face mesh setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# === Haar cascades ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# === Preprocessing ===
def preprocess_gaze(eye):
    eye = cv2.resize(eye, IMG_GAZE_SIZE)
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    eye = (eye - 0.5) / 0.5
    return np.transpose(eye, (2, 0, 1))[np.newaxis, :]

def preprocess_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_FACE_SIZE)
    face_tensor = face_resized / 255.0
    face_tensor = (face_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return np.transpose(face_tensor.astype(np.float32), (2, 0, 1))[np.newaxis, :]

def estimate_head_position(landmarks):
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]
    top = landmarks[10]
    bottom = landmarks[152]
    if nose.x < left.x:
        return "turned left"
    elif nose.x > right.x:
        return "turned right"
    elif nose.y < top.y:
        return "tilted up"
    elif nose.y > bottom.y:
        return "tilted down"
    return "center"

def estimate_drowsiness(blink_rate, gaze):
    return "drowsy" if blink_rate > 25 or gaze == "down" else "awake"

def estimate_cognitive_load(blink_rate, emotion):
    if blink_rate > 25 or emotion in NEGATIVE_EMOTIONS:
        return "high"
    elif emotion == "neutral" and blink_rate < 10:
        return "low"
    else:
        return "medium"

# === Action Triggers ===
def minimize_all_windows():
    for win in getWindowsWithTitle(''):
        try: win.minimize()
        except: continue

def play_calming_music():
    def _play():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load("calm.wav")
            pygame.mixer.music.play(-1)
        except Exception as e:
            print("[ERROR] Music:", e)
    threading.Thread(target=_play, daemon=True).start()

def stop_music():
    try:
        pygame.mixer.music.stop()
    except:
        pass

def reduce_brightness():
    try:
        sbc.set_brightness(30)
    except:
        pass

def lock_workstation():
    ctypes.windll.user32.LockWorkStation()

# === Blink tracking ===
TOP_LID, BOTTOM_LID = 159, 145
blink_count = 0
prev_eye_state = "OPEN"
blink_log_start = time.time()
distracted_start = None
sad_start = None
high_load_start = None
presence_start = time.time()
no_face_start = None

# === Emotion timer ===
emotion_timer_start = None
last_emotion = None
emotion_duration = 0

# === Webcam start ===
cap = cv2.VideoCapture(0)
print("[INFO] Monitoring started. Press 'q' to quit or 's' to stop music.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_detected = False
        eye_state = "OPEN"
        lid_distance = 10.0
        head_position = "Unknown"
        face_emotion = "neutral"
        face_conf = 0.0
        gaze_result = "Unknown"

        if results.multi_face_landmarks:
            face_detected = True
            no_face_start = None
            landmarks = results.multi_face_landmarks[0].landmark

            # Blink logic
            top = landmarks[TOP_LID]
            bottom = landmarks[BOTTOM_LID]
            lid_distance = np.linalg.norm(
                np.array([top.x * w, top.y * h]) - np.array([bottom.x * w, bottom.y * h])
            )
            eye_state = "CLOSED" if lid_distance < CLOSED_THRESHOLD else "OPEN"
            if prev_eye_state == "OPEN" and eye_state == "CLOSED":
                blink_start = True
            elif prev_eye_state == "CLOSED" and eye_state == "OPEN":
                blink_count += 1
            prev_eye_state = eye_state

            head_position = estimate_head_position(landmarks)

            # Face detection (OpenCV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, fw, fh) = faces[0]
                face_img = frame[y:y+fh, x:x+fw]

                eyes = eye_cascade.detectMultiScale(face_img)
                if len(eyes) > 0:
                    (ex, ey, ew, eh) = eyes[0]
                    eye_img = face_img[ey:ey+eh, ex:ex+ew]
                    gaze_tensor = preprocess_gaze(eye_img)
                    gaze_out = sess_gaze.run([output_gaze], {input_gaze: gaze_tensor})[0]
                    gaze_result = LABELS_GAZE[int(np.argmax(gaze_out))]

                face_tensor = preprocess_face(face_img)
                face_out = sess_face.run([output_face], {input_face: face_tensor})[0]
                face_emotion = LABELS_EMOTION[int(np.argmax(face_out))]
                face_conf = float(np.max(face_out))

                # Emotion tracking
                if face_emotion != last_emotion:
                    emotion_timer_start = time.time()
                    last_emotion = face_emotion
                    emotion_duration = 0
                elif emotion_timer_start:
                    emotion_duration = int(time.time() - emotion_timer_start)

        else:
            if no_face_start is None:
                no_face_start = time.time()
            elif time.time() - no_face_start > 60:
                lock_workstation()
            cv2.putText(frame, "NO FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Unified AI Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stop_music()
            continue

        # === State Estimations ===
        minutes_passed = (time.time() - blink_log_start) / 60.0
        bpm = blink_count / minutes_passed if minutes_passed > 0 else 0
        drowsy_state = estimate_drowsiness(bpm, gaze_result)
        cog_load = estimate_cognitive_load(bpm, face_emotion)

        # === Triggers ===
        now = time.time()
        if gaze_result != "straight":
            if distracted_start is None:
                distracted_start = now
            elif now - distracted_start > 20:
                minimize_all_windows()
                distracted_start = None
        else:
            distracted_start = None

        if face_emotion == "sadness":
            if sad_start is None:
                sad_start = now
            elif now - sad_start > 60:
                play_calming_music()
                sad_start = None
        else:
            sad_start = None

        if cog_load == "high":
            if high_load_start is None:
                high_load_start = now
            elif now - high_load_start > 90:
                reduce_brightness()
                high_load_start = None
        else:
            high_load_start = None

        if now - presence_start > 2400:
            print("[INFO] Suggest a 5-min break!")
            presence_start = now

        # === Display Emotion + Duration ===
        emotion_text = f"{face_emotion.upper()} ({emotion_duration}s)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (w - text_size[0] - 20, 10), (w - 10, 10 + text_size[1] + 10), (0, 0, 0), -1)
        cv2.putText(frame, emotion_text, (w - text_size[0] - 15, 10 + text_size[1] + 5), font, font_scale, (0, 255, 255), font_thickness)

        # === Save to JSON ===
        state = {
            "face": "yes",
            "face_emotion": f"{face_emotion} ({face_conf:.2f})",
            "emotion_duration_sec": emotion_duration,
            "blinking_rate_BPM": round(bpm, 2),
            "gaze": gaze_result,
            "head_position": head_position,
            "drowsy_state": drowsy_state,
            "cognitive_load": cog_load,
            "timestamp": time.strftime("%H:%M:%S")
        }
        with open("realtime_log.json", "w") as f:
            json.dump(state, f, indent=2)
        print(json.dumps(state, indent=2))

        # === Final Dashboard Summary ===
        active_minutes = int((time.time() - presence_start) / 60)
        mood_trend = "Downtrend" if face_emotion in NEGATIVE_EMOTIONS else "Stable"
        pitch = round((landmarks[10].y - landmarks[152].y) * 90, 1)
        yaw = round((landmarks[234].x - landmarks[454].x) * 90, 1)
        roll = 0.0
        suggested_action = "Suggested break" if gaze_result != "straight" or drowsy_state == "drowsy" or cog_load == "high" else "None"
        if suggested_action != "None":
            print(f"[{time.strftime('%H:%M:%S')}] Action: {suggested_action} – User looking away for 30s")

        print(f"""
Face Presence     : {'✅ Present' if face_detected else '❌ Absent'}
Gaze              : 👁️ Looking {'at screen' if gaze_result == 'straight' else gaze_result}
Emotion           : 😰 {face_emotion.capitalize()} (Confidence: {int(face_conf * 100)}%)
Mood Stability    : {mood_trend}
Cognitive Load    : {cog_load.upper()}
Blink Rate        : {int(bpm)} BPM
Head Pose         : Pitch {pitch}°, Yaw {yaw}°, Roll {roll}°
Active Time       : {active_minutes} minutes
System Action     : {suggested_action}
----------------------------------------------
""")

        cv2.imshow("Unified AI Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stop_music()

except KeyboardInterrupt:
    print("[INFO] Stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    stop_music()
