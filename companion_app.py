"""
ISL Connect - Polished Companion Middleware
- No extra OpenCV window
- Cleaner status handling
- Live ISL → Caption
- Speech → Avatar
"""

import tkinter as tk
from tkinter import ttk
import threading
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque, Counter
import json
from pathlib import Path
import speech_recognition as sr

from src.models.transformer import TransformerClassifier
from src.utils.config import load_config, get_device
from src.text_to_isl.sign_mapper import SignMapper
from src.text_to_isl.skeleton_avatar import SkeletonAvatar


class CompanionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Connect")
        self.root.geometry("430x400")
        self.root.configure(bg="#11111b")
        self.root.attributes("-topmost", True)
        self.root.minsize(380, 360)

        # ---------- Model ----------
        self.cfg = load_config()
        self.device = get_device(False)
        ckpt_path = Path(self.cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = TransformerClassifier(
            input_dim=300,
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_feedforward=256,
            num_classes=ckpt["num_classes"],
            dropout=0.2
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        with open(self.cfg["paths"]["label_map"]) as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmark_buffer = deque(maxlen=45)
        self.recent_preds = deque(maxlen=3)
        self.current_stable = None
        self.camera_running = False

        self.mapper = SignMapper()
        self.avatar = SkeletonAvatar()
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.history = []

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#11111b")
        header.pack(fill="x", padx=15, pady=(12, 4))

        tk.Label(
            header, text="ISL Connect",
            font=("Segoe UI", 18, "bold"),
            fg="#89b4fa", bg="#11111b"
        ).pack(side="left")

        self.mode_var = tk.StringVar(value="Call Mode")
        ttk.Combobox(
            header, textvariable=self.mode_var,
            values=["Call Mode", "Content Mode"],
            state="readonly", width=12, font=("Segoe UI", 9)
        ).pack(side="right")

        self.status_var = tk.StringVar(value="● Ready")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=("Segoe UI", 10),
            fg="#a6e3a1", bg="#11111b"
        ).pack(pady=(2, 6))

        # Caption
        caption_frame = tk.Frame(self.root, bg="#1e1e2e", padx=12, pady=12)
        caption_frame.pack(fill="both", expand=True, padx=15, pady=4)

        tk.Label(
            caption_frame, text="LIVE CAPTION",
            font=("Segoe UI", 8, "bold"),
            fg="#6c7086", bg="#1e1e2e"
        ).pack(anchor="w")

        self.caption_var = tk.StringVar(value="Waiting for input...")
        tk.Label(
            caption_frame, textvariable=self.caption_var,
            font=("Segoe UI", 14, "bold"),
            fg="#f9e2af", bg="#1e1e2e",
            wraplength=370, justify="left", anchor="nw"
        ).pack(fill="both", expand=True, pady=(6, 0))

        # History
        self.history_var = tk.StringVar(value="")
        tk.Label(
            self.root, textvariable=self.history_var,
            font=("Segoe UI", 8),
            fg="#6c7086", bg="#11111b",
            wraplength=390, justify="left"
        ).pack(padx=15, anchor="w")

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#11111b")
        btn_frame.pack(pady=12)

        self.btn_camera = tk.Button(
            btn_frame, text="📷  Camera",
            font=("Segoe UI", 10, "bold"),
            bg="#89b4fa", fg="#11111b",
            relief="flat", padx=12, pady=7,
            command=self.toggle_camera
        )
        self.btn_camera.grid(row=0, column=0, padx=5)

        self.btn_speak = tk.Button(
            btn_frame, text="🎤  Speak",
            font=("Segoe UI", 10, "bold"),
            bg="#a6e3a1", fg="#11111b",
            relief="flat", padx=12, pady=7,
            command=self.on_speak
        )
        self.btn_speak.grid(row=0, column=1, padx=5)

        self.btn_clear = tk.Button(
            btn_frame, text="Clear",
            font=("Segoe UI", 10),
            bg="#313244", fg="white",
            relief="flat", padx=12, pady=7,
            command=self.clear
        )
        self.btn_clear.grid(row=0, column=2, padx=5)

        tk.Label(
            self.root,
            text="Pin this window on top of Zoom / Meet / WhatsApp",
            font=("Segoe UI", 8),
            fg="#585b70", bg="#11111b"
        ).pack(side="bottom", pady=8)

    def clear(self):
        self.caption_var.set("Waiting for input...")
        self.status_var.set("● Ready")
        self.history_var.set("")
        self.history = []
        self.current_stable = None

    def add_to_history(self, text):
        self.history.append(text)
        if len(self.history) > 3:
            self.history.pop(0)
        self.history_var.set("Recent: " + "  →  ".join(self.history))

    def toggle_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.btn_camera.config(text="⏹  Stop", bg="#f38ba8")
            self.status_var.set("● Camera active (sign now)")
            threading.Thread(target=self._camera_loop, daemon=True).start()
        else:
            self.camera_running = False
            self.btn_camera.config(text="📷  Camera", bg="#89b4fa")
            self.status_var.set("● Camera stopped")

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            gloss = self._process_frame(frame)
            if gloss:
                display = gloss.replace("_", " ").title()
                self.root.after(0, lambda g=display: self.caption_var.set(g))
                self.root.after(0, lambda g=display: self.add_to_history(g))
        cap.release()
        self.root.after(0, lambda: self.btn_camera.config(text="📷  Camera", bg="#89b4fa"))
        self.root.after(0, lambda: self.status_var.set("● Ready"))

    def _process_frame(self, frame):
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

        self.landmark_buffer.append(lm.flatten())
        if len(self.landmark_buffer) < 45:
            return None

        seq = np.array(self.landmark_buffer, dtype=np.float32)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        if conf.item() < 0.52:
            self.recent_preds.append(None)
            return None

        gloss = self.inv_label_map.get(pred_idx.item(), None)
        self.recent_preds.append(gloss)

        if len(self.recent_preds) == 3:
            valid = [p for p in self.recent_preds if p is not None]
            if valid:
                most_common, count = Counter(valid).most_common(1)[0]
                if count >= 2 and most_common != self.current_stable:
                    self.current_stable = most_common
                    return most_common
        return None

    def on_speak(self):
        if self.listening:
            return
        self.listening = True
        self.status_var.set("● Listening...")
        self.btn_speak.config(state="disabled")

        def task():
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, lambda: self._handle_speech(text))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self._on_error(msg))

        threading.Thread(target=task, daemon=True).start()

    def _handle_speech(self, text):
        self.caption_var.set(text)
        self.add_to_history(text)
        glosses = self.mapper.text_to_glosses(text)

        if glosses:
            self.caption_var.set(" → ".join(glosses))
            self.status_var.set("● Avatar signing...")
            def play():
                self.avatar.play_sequence(glosses)
                self.root.after(0, lambda: self.status_var.set("● Ready"))
                self.root.after(0, lambda: self.btn_speak.config(state="normal"))
                self.listening = False
            threading.Thread(target=play, daemon=True).start()
        else:
            self.status_var.set("● Unknown phrase")
            self.btn_speak.config(state="normal")
            self.listening = False

    def _on_error(self, msg):
        self.status_var.set(f"● {msg}")
        self.btn_speak.config(state="normal")
        self.listening = False


if __name__ == "__main__":
    root = tk.Tk()
    app = CompanionApp(root)
    root.mainloop()