"""
Real-time ISL Predictor
Webcam → MediaPipe → Landmarks → Buffer → Model → Stabilization → Text + TTS
"""

import time
import json
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.models import GRUClassifier, TCNClassifier, HybridTCNGRU
from src.data.normalize import LandmarkNormalizer, flatten_landmarks
from src.inference.smoothing import PredictionStabilizer
from src.inference.tts import TTSEngine
from src.utils.config import load_config, get_device
from src.utils.logger import setup_logger


class RealTimePredictor:
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        checkpoint_path: str = None,
        label_map_path: str = None,
    ):
        self.cfg = load_config(config_path)
        self.logger = setup_logger("realtime")

        self.device = get_device(self.cfg["training"]["device"] == "cuda")
        self.window_size = self.cfg["inference"]["window_size"]
        self.stride = self.cfg["inference"]["stride"]

        # Paths
        ckpt_path = checkpoint_path or self.cfg["paths"]["best_model"]
        label_path = label_map_path or self.cfg["paths"]["label_map"]

        # Label map
        with open(label_path, "r") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)

        # Model
        self.model = self._load_model(ckpt_path)
        self.model.eval()

        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Normalization
        self.normalizer = LandmarkNormalizer()

        # Temporal buffer
        self.buffer = deque(maxlen=self.cfg["inference"]["max_buffer_size"])
        self.frame_count = 0

        # Stabilizer
        self.stabilizer = PredictionStabilizer(
            num_classes=self.num_classes,
            confidence_threshold=self.cfg["inference"]["confidence_threshold"],
            window_size=self.cfg["inference"]["stabilization_window"],
            min_stable_frames=self.cfg["inference"]["min_stable_frames"],
            cooldown_frames=self.cfg["inference"]["cooldown_frames"],
            no_sign_threshold=self.cfg["inference"]["no_sign_threshold"],
        )

        # TTS
        self.tts = TTSEngine(
            rate=self.cfg["tts"]["rate"],
            volume=self.cfg["tts"]["volume"],
            cooldown_seconds=self.cfg["tts"]["cooldown_seconds"],
            enabled=self.cfg["tts"]["enabled"],
        )

        # State
        self.current_prediction = "—"
        self.current_confidence = 0.0
        self.last_inference_time = 0.0

        self.logger.info("RealTimePredictor initialized successfully")

    def _load_model(self, ckpt_path: str) -> torch.nn.Module:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_name = ckpt.get("model_name", self.cfg["model"]["name"])
        num_classes = ckpt.get("num_classes", self.num_classes)

        if model_name == "gru":
            model = GRUClassifier(
                input_dim=300,
                hidden_dim=self.cfg["model"]["hidden_dim"],
                num_layers=self.cfg["model"].get("num_layers", 2),
                num_classes=num_classes,
                dropout=0.0,
            )
        elif model_name == "tcn":
            model = TCNClassifier(
                input_dim=300,
                num_channels=[64, 128, 128, 256],
                kernel_size=3,
                dropout=0.0,
                num_classes=num_classes,
            )
        elif model_name == "hybrid":
            model = HybridTCNGRU(
                input_dim=300,
                num_classes=num_classes,
                dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown model in checkpoint: {model_name}")

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.logger.info(f"Loaded {model_name} model from {ckpt_path}")
        return model

    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract 75×4 landmarks from MediaPipe Holistic results."""
        landmarks = np.zeros((75, 4), dtype=np.float32)

        # Pose (33)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        # Left hand (21) → indices 33–53
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                landmarks[33 + i] = [lm.x, lm.y, lm.z, 1.0]

        # Right hand (21) → indices 54–74
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                landmarks[54 + i] = [lm.x, lm.y, lm.z, 1.0]

        return landmarks

    @torch.no_grad()
    def _run_inference(self) -> Tuple[Optional[str], float]:
        """Run model on the current temporal window."""
        if len(self.buffer) < self.window_size:
            return None, 0.0

        # Take the most recent window
        window = list(self.buffer)[-self.window_size:]
        seq = np.stack(window)                     # (T, 75, 4)
        seq = self.normalizer(seq)
        seq = flatten_landmarks(seq)               # (T, 300)
        seq = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, 300)

        logits = self.model(seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx, conf, is_new = self.stabilizer.update(probs)

        if pred_idx is not None and pred_idx in self.inv_label_map:
            gloss = self.inv_label_map[pred_idx]
            if is_new:
                self.tts.speak(gloss.replace("_", " ").lower())
            return gloss, conf

        return None, conf

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame and return annotated frame.
        """
        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        # Extract & buffer landmarks
        landmarks = self._extract_landmarks(results)
        if landmarks is not None:
            self.buffer.append(landmarks)

        # Inference every `stride` frames
        if self.frame_count % self.stride == 0:
            t0 = time.time()
            pred, conf = self._run_inference()
            self.last_inference_time = (time.time() - t0) * 1000  # ms

            if pred is not None:
                self.current_prediction = pred
                self.current_confidence = conf

        # Draw landmarks
        annotated = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            )

        # Overlay UI
        self._draw_ui(annotated)
        return annotated

    def _draw_ui(self, frame: np.ndarray):
        h, w = frame.shape[:2]

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "LIVE ISL TRANSLATOR  |  Phase 1", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Prediction : {self.current_prediction}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence : {self.current_confidence*100:.1f}%", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Latency
        cv2.putText(frame, f"Inference: {self.last_inference_time:.1f} ms", (w - 260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1)

    def run_webcam(self, camera_id: int = 0):
        """Main real-time loop."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.logger.info("Starting webcam inference. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self.process_frame(frame)
                cv2.imshow("ISL Real-time Translator", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            self.tts.stop()
            self.logger.info("Webcam session ended.")


if __name__ == "__main__":
    predictor = RealTimePredictor()
    predictor.run_webcam()
