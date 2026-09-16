"""
Real-time ISL → Full Sentence (Debug + Improved version)
"""

import cv2
import time
import numpy as np
import torch
import mediapipe as mp
from collections import deque, Counter
import pyttsx3
import threading
from pathlib import Path
import json

from src.models import TCNClassifier
from src.utils.config import load_config, get_device


# ====================== Tunable Parameters ======================
WINDOW_SIZE = 45
CONFIDENCE_THRESHOLD = 0.45      # lowered
STABLE_NEEDED = 3                # lowered
PAUSE_SECONDS = 1.6
COOLDOWN_AFTER_SPEAK = 2.0


class SentenceBuilder:
    def __init__(self):
        self.templates = {
            ("HELLO",): "Hello.",
            ("THANK_YOU",): "Thank you.",
            ("HOW_ARE_YOU",): "How are you?",
            ("GOOD_MORNING",): "Good morning.",
            ("GOOD_AFTERNOON",): "Good afternoon.",
            ("GOOD_EVENING",): "Good evening.",
            ("GOOD_NIGHT",): "Good night.",
            ("ALRIGHT",): "Alright.",
            ("PLEASED",): "Pleased to meet you.",
            ("HELLO", "HOW_ARE_YOU"): "Hello, how are you?",
            ("GOOD_MORNING", "HOW_ARE_YOU"): "Good morning, how are you?",
            ("HELLO", "THANK_YOU"): "Hello, thank you.",
            ("THANK_YOU", "HELLO"): "Thank you.",
        }

    def build(self, glosses):
        if not glosses:
            return ""
        key = tuple(glosses)
        if key in self.templates:
            return self.templates[key]
        if len(glosses) >= 2:
            key2 = tuple(glosses[:2])
            if key2 in self.templates:
                return self.templates[key2]
        # Fallback
        return " ".join(g.replace("_", " ").title() for g in glosses) + "."


class RealTimeSentencePredictor:
    def __init__(self):
        self.cfg = load_config()
        self.device = get_device(False)

        # Load model
        ckpt_path = Path(self.cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
        print(f"Loading model from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = TCNClassifier(
            input_dim=300,
            num_channels=[64, 128, 128, 256],
            kernel_size=3,
            dropout=0.2,
            num_classes=ckpt["num_classes"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        with open(self.cfg["paths"]["label_map"]) as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        print("Classes:", list(self.inv_label_map.values()))

        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmark_buffer = deque(maxlen=WINDOW_SIZE)
        self.recent_preds = deque(maxlen=STABLE_NEEDED)
        self.sentence_glosses = []
        self.last_detection_time = time.time()
        self.last_speak_time = 0
        self.current_stable = None

        # TTS setup
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 145)
        self.tts_lock = threading.Lock()
        self.sentence_builder = SentenceBuilder()

        print("\nSystem ready!")
        print("Sign some words, then pause for 1.5–2 seconds...")
        print("Press 'q' to quit\n")

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        lm = np.zeros((75, 4), dtype=np.float32)

        if results.pose_landmarks:
            for i, p in enumerate(results.pose_landmarks.landmark):
                lm[i] = [p.x, p.y, p.z, p.visibility]
        if results.left_hand_landmarks:
            for i, p in enumerate(results.left_hand_landmarks.landmark):
                lm[33 + i] = [p.x, p.y, p.z, 1.0]
        if results.right_hand_landmarks:
            for i, p in enumerate(results.right_hand_landmarks.landmark):
                lm[54 + i] = [p.x, p.y, p.z, 1.0]
        return lm.flatten()

    def speak(self, text):
        print(f"Speaking: {text}")
        def _speak():
            try:
                with self.tts_lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                print("TTS Error:", e)
        threading.Thread(target=_speak, daemon=True).start()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.extract_landmarks(frame)
            self.landmark_buffer.append(landmarks)

            gloss = None
            conf = 0.0

            if len(self.landmark_buffer) >= WINDOW_SIZE:
                seq = np.array(self.landmark_buffer, dtype=np.float32)
                x = torch.from_numpy(seq).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    conf = conf.item()
                    gloss = self.inv_label_map.get(pred_idx.item(), "UNKNOWN")

                # Stabilization
                if conf >= CONFIDENCE_THRESHOLD:
                    self.recent_preds.append(gloss)
                else:
                    self.recent_preds.append(None)

                if len(self.recent_preds) == STABLE_NEEDED:
                    valid = [p for p in self.recent_preds if p is not None]
                    if valid:
                        most_common, count = Counter(valid).most_common(1)[0]
                        if count >= STABLE_NEEDED - 1:
                            if most_common != self.current_stable:
                                self.current_stable = most_common
                                self.sentence_glosses.append(most_common)
                                self.last_detection_time = time.time()
                                print(f"→ Added to sentence: {most_common}")

            # Pause detection → speak sentence
            if (self.sentence_glosses and 
                time.time() - self.last_detection_time > PAUSE_SECONDS and
                time.time() - self.last_speak_time > COOLDOWN_AFTER_SPEAK):

                sentence = self.sentence_builder.build(self.sentence_glosses)
                print(f"\n>>> FULL SENTENCE: {sentence}\n")
                self.speak(sentence)
                self.last_speak_time = time.time()
                self.sentence_glosses = []
                self.current_stable = None
                self.recent_preds.clear()

            # ====== UI ======
            cv2.putText(frame, f"Current: {self.current_stable or '-'}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            sent_text = " → ".join(self.sentence_glosses) if self.sentence_glosses else "-"
            cv2.putText(frame, f"Building: {sent_text}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 50), 2)

            cv2.imshow("ISL Sentence Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


if __name__ == "__main__":
    predictor = RealTimeSentencePredictor()
    predictor.run()