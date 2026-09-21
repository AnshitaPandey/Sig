"""
Final Bidirectional ISL System
- Live ISL → Text + Speech  (Webcam)
- Speech → ISL Avatar       (Microphone)
Both directions work in parallel.
"""

import cv2
import time
import threading
import numpy as np
import torch
import mediapipe as mp
from collections import deque, Counter
import pyttsx3
import speech_recognition as sr
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from src.models.transformer import TransformerClassifier   # or TCNClassifier
from src.utils.config import load_config, get_device
from src.text_to_isl.sign_mapper import SignMapper
from src.text_to_isl.skeleton_avatar import SkeletonAvatar


# ====================== CONFIG ======================
WINDOW_SIZE = 45
CONF_THRESHOLD = 0.48
STABLE_NEEDED = 3
PAUSE_SECONDS = 1.7
COOLDOWN = 2.2


class BidirectionalISL:
    def __init__(self):
        self.cfg = load_config()
        self.device = get_device(False)

        # ---------- Load Model ----------
        ckpt_path = Path(self.cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Load the architecture that matches the saved checkpoint
        self.model = TransformerClassifier(
            input_dim=300,
            d_model=128,               # must match checkpoint
            nhead=4,
            num_layers=3,              # must match checkpoint
            dim_feedforward=256,
            num_classes=ckpt["num_classes"],
            dropout=0.2
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print("Model loaded successfully!")

        with open(self.cfg["paths"]["label_map"]) as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # ---------- MediaPipe ----------
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ---------- Buffers ----------
        self.landmark_buffer = deque(maxlen=WINDOW_SIZE)
        self.recent_preds = deque(maxlen=STABLE_NEEDED)
        self.sentence_glosses = []
        self.last_detection_time = time.time()
        self.last_speak_time = 0
        self.current_stable = None

        # ---------- TTS ----------
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 145)
        self.tts_lock = threading.Lock()

        # ---------- Reverse direction ----------
        self.mapper = SignMapper()
        self.avatar = SkeletonAvatar()
        self.recognizer = sr.Recognizer()

        self.running = True
        self.listening = False

    def speak(self, text):
        def _s():
            with self.tts_lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=_s, daemon=True).start()

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        lm = np.zeros((75, 4), dtype=np.float32)

        if results.pose_landmarks:
            for i, p in enumerate(results.pose_landmarks.landmark):
                lm[i] = [p.x, p.y, p.z, p.visibility]
        if results.left_hand_landmarks:
            for i, p in enumerate(results.left_hand_landmarks.landmark):
                lm[33+i] = [p.x, p.y, p.z, 1.0]
        if results.right_hand_landmarks:
            for i, p in enumerate(results.right_hand_landmarks.landmark):
                lm[54+i] = [p.x, p.y, p.z, 1.0]
        return lm.flatten()

    def process_frame(self, frame):
        landmarks = self.extract_landmarks(frame)
        self.landmark_buffer.append(landmarks)

        gloss, conf = None, 0.0

        if len(self.landmark_buffer) >= WINDOW_SIZE:
            seq = np.array(self.landmark_buffer, dtype=np.float32)
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
                conf = conf.item()
                gloss = self.inv_label_map.get(pred_idx.item(), "UNKNOWN")

            if conf >= CONF_THRESHOLD:
                self.recent_preds.append(gloss)
            else:
                self.recent_preds.append(None)

            if len(self.recent_preds) == STABLE_NEEDED:
                valid = [p for p in self.recent_preds if p is not None]
                if valid:
                    most_common, count = Counter(valid).most_common(1)[0]
                    if count >= STABLE_NEEDED - 1 and most_common != self.current_stable:
                        self.current_stable = most_common
                        self.sentence_glosses.append(most_common)
                        self.last_detection_time = time.time()

        # Sentence finished?
        if (self.sentence_glosses and
            time.time() - self.last_detection_time > PAUSE_SECONDS and
            time.time() - self.last_speak_time > COOLDOWN):

            sentence = " ".join(g.replace("_", " ").title() for g in self.sentence_glosses)
            print(f"\n>>> ISL → Speech: {sentence}")
            self.speak(sentence)
            self.last_speak_time = time.time()
            self.sentence_glosses = []
            self.current_stable = None
            self.recent_preds.clear()

        return self.current_stable, conf

    def listen_and_sign(self):
        """Speech → Avatar (runs in background)"""
        self.listening = True
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=5)
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")

            glosses = self.mapper.text_to_glosses(text)
            if glosses:
                print(f"Avatar signing: {' → '.join(glosses)}")
                self.avatar.play_sequence(glosses)
            else:
                print("Unknown phrase")
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.listening = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("\n" + "="*60)
        print("  BIDIRECTIONAL ISL SYSTEM STARTED")
        print("  - Sign in front of camera → System speaks")
        print("  - Press 's' to speak → Avatar signs back")
        print("  - Press 'q' to quit")
        print("="*60 + "\n")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gloss, conf = self.process_frame(frame)

            # UI overlay
            cv2.putText(frame, f"Current: {gloss or '-'}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            sent = " → ".join(self.sentence_glosses) if self.sentence_glosses else "-"
            cv2.putText(frame, f"Building: {sent}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)

            status = "Listening..." if self.listening else "Press 's' to speak"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.imshow("Bidirectional ISL System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s') and not self.listening:
                threading.Thread(target=self.listen_and_sign, daemon=True).start()

        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


if __name__ == "__main__":
    system = BidirectionalISL()
    system.run()