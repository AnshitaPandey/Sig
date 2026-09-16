"""
Prediction Stabilization for Real-time ISL Recognition.
Prevents flickering predictions (HELP → HELLO → HELP ...).
"""

from collections import deque
from typing import Optional, Tuple
import numpy as np


class PredictionStabilizer:
    """
    Multi-strategy temporal stabilization.

    Techniques used:
    - Confidence threshold
    - Majority voting over a sliding window
    - Minimum consecutive stable frames
    - Cooldown after a spoken prediction
    - Explicit NO_SIGN / REST handling
    - Duplicate suppression
    """

    def __init__(
        self,
        num_classes: int,
        confidence_threshold: float = 0.65,
        window_size: int = 7,
        min_stable_frames: int = 4,
        cooldown_frames: int = 15,
        no_sign_class: int = -1,          # special index for NO_SIGN
        no_sign_threshold: float = 0.40,
    ):
        self.num_classes = num_classes
        self.conf_thresh = confidence_threshold
        self.window_size = window_size
        self.min_stable = min_stable_frames
        self.cooldown = cooldown_frames
        self.no_sign_class = no_sign_class
        self.no_sign_thresh = no_sign_threshold

        self.history = deque(maxlen=window_size)
        self.cooldown_counter = 0
        self.last_stable_pred = None
        self.stable_count = 0

    def update(
        self,
        probs: np.ndarray,
    ) -> Tuple[Optional[int], float, bool]:
        """
        Args:
            probs: softmax probabilities of shape (num_classes,)

        Returns:
            pred_idx: stabilized class index or None
            confidence: confidence of the decision
            is_new: True if this is a newly stabilized prediction (ready for TTS)
        """
        # Cooldown after speaking
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None, 0.0, False

        max_prob = float(np.max(probs))
        pred = int(np.argmax(probs))

        # Low confidence → treat as NO_SIGN / REST
        if max_prob < self.no_sign_thresh:
            self.history.append(self.no_sign_class)
            self.stable_count = 0
            self.last_stable_pred = None
            return None, max_prob, False

        if max_prob < self.conf_thresh:
            self.history.append(self.no_sign_class)
            return None, max_prob, False

        self.history.append(pred)

        # Need enough history
        if len(self.history) < self.min_stable:
            return None, max_prob, False

        # Majority vote
        recent = list(self.history)[-self.min_stable:]
        values, counts = np.unique(recent, return_counts=True)
        majority_idx = values[np.argmax(counts)]
        majority_count = counts.max()

        if majority_idx == self.no_sign_class:
            self.stable_count = 0
            self.last_stable_pred = None
            return None, max_prob, False

        if majority_count >= self.min_stable:
            # Stable prediction found
            is_new = (majority_idx != self.last_stable_pred)

            if is_new:
                self.last_stable_pred = majority_idx
                self.cooldown_counter = self.cooldown
                self.stable_count = 0
                return majority_idx, max_prob, True
            else:
                # Same as previous – suppress duplicate
                return majority_idx, max_prob, False

        return None, max_prob, False

    def reset(self):
        self.history.clear()
        self.cooldown_counter = 0
        self.last_stable_pred = None
        self.stable_count = 0
