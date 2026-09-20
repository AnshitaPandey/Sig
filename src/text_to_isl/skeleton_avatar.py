"""
Skeleton Avatar Player using MediaPipe-style landmarks
"""

import cv2
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

# MediaPipe Pose connections (simplified)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32)
]

# Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


class SkeletonAvatar:
    def __init__(self, landmarks_dir="data/processed/landmarks"):
        self.landmarks_dir = Path(landmarks_dir)
        self.gloss_to_sequences = self._load_sequences()
        print(f"Loaded skeleton sequences for {len(self.gloss_to_sequences)} glosses.")

    def _load_sequences(self):
        data = {}
        if not self.landmarks_dir.exists():
            print("Warning: landmarks directory not found!")
            return data

        for class_dir in self.landmarks_dir.iterdir():
            if class_dir.is_dir():
                gloss = class_dir.name.upper()
                npy_files = list(class_dir.glob("*.npy"))
                if npy_files:
                    # Load the first available sequence for this gloss
                    seq = np.load(npy_files[0])  # shape: (T, 75, 4)
                    data[gloss] = seq
        return data

    def _draw_landmarks(self, canvas, landmarks, color=(0, 255, 0), radius=4):
        """
        landmarks: (75, 4) → pose (0-32) + left hand (33-53) + right hand (54-74)
        """
        h, w = canvas.shape[:2]

        def to_pixel(lm):
            x, y = int(lm[0] * w), int(lm[1] * h)
            return x, y

        # Draw Pose
        pose = landmarks[:33]
        for i, j in POSE_CONNECTIONS:
            if pose[i][3] > 0.3 and pose[j][3] > 0.3:  # visibility check
                pt1 = to_pixel(pose[i])
                pt2 = to_pixel(pose[j])
                cv2.line(canvas, pt1, pt2, color, 2)
        for lm in pose:
            if lm[3] > 0.3:
                cv2.circle(canvas, to_pixel(lm), radius, color, -1)

        # Draw Left Hand
        left_hand = landmarks[33:54]
        for i, j in HAND_CONNECTIONS:
            pt1 = to_pixel(left_hand[i])
            pt2 = to_pixel(left_hand[j])
            cv2.line(canvas, pt1, pt2, (255, 128, 0), 2)
        for lm in left_hand:
            cv2.circle(canvas, to_pixel(lm), 3, (255, 128, 0), -1)

        # Draw Right Hand
        right_hand = landmarks[54:75]
        for i, j in HAND_CONNECTIONS:
            pt1 = to_pixel(right_hand[i])
            pt2 = to_pixel(right_hand[j])
            cv2.line(canvas, pt1, pt2, (0, 128, 255), 2)
        for lm in right_hand:
            cv2.circle(canvas, to_pixel(lm), 3, (0, 128, 255), -1)

        return canvas

    def play_gloss(self, gloss, canvas_size=(720, 540), fps=25):
        gloss = gloss.upper()
        seq = self.gloss_to_sequences.get(gloss)
        if seq is None:
            print(f"No skeleton data for: {gloss}")
            return False

        print(f"Playing skeleton: {gloss} ({len(seq)} frames)")

        for frame_lm in seq:
            canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 30  # dark background
            canvas = self._draw_landmarks(canvas, frame_lm)

            # Add text
            cv2.putText(canvas, f"ISL Avatar: {gloss}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2)

            cv2.imshow("ISL Skeleton Avatar", canvas)
            key = cv2.waitKey(int(1000 / fps))
            if key == ord('q'):
                break

        return True

    def play_sequence(self, glosses, pause_between=0.5):
        for gloss in glosses:
            self.play_gloss(gloss)
            time.sleep(pause_between)
        cv2.destroyWindow("ISL Skeleton Avatar")