"""
Text → ISL Gloss → Sign Mapping
"""

from pathlib import Path
import json

class SignMapper:
    def __init__(self, signs_dir="data/signs"):
        self.signs_dir = Path(signs_dir)
        self.signs_dir.mkdir(parents=True, exist_ok=True)

        # Mapping from English words / phrases to our glosses
        self.word_to_gloss = {
            "hello": "HELLO",
            "hi": "HELLO",
            "thank you": "THANK_YOU",
            "thanks": "THANK_YOU",
            "how are you": "HOW_ARE_YOU",
            "good morning": "GOOD_MORNING",
            "good afternoon": "GOOD_AFTERNOON",
            "good evening": "GOOD_EVENING",
            "good night": "GOOD_NIGHT",
            "alright": "ALRIGHT",
            "ok": "ALRIGHT",
            "okay": "ALRIGHT",
            "pleased": "PLEASED",
            "dog": "DOG",
            "cat": "CAT",
            "fish": "FISH",
            "bird": "BIRD",
            "cow": "COW",
            "mouse": "MOUSE",
            "horse": "HORSE",
            "animal": "ANIMAL",
        }

        # Reverse mapping (for later use)
        self.gloss_to_word = {v: k for k, v in self.word_to_gloss.items()}

    def text_to_glosses(self, text: str) -> list:
        """
        Convert English text into a list of ISL glosses.
        Simple rule-based for now (can be upgraded later).
        """
        text = text.lower().strip()
        glosses = []

        # Check multi-word phrases first
        for phrase, gloss in sorted(self.word_to_gloss.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                glosses.append(gloss)
                text = text.replace(phrase, "")

        # Then single words
        for word in text.split():
            if word in self.word_to_gloss:
                glosses.append(self.word_to_gloss[word])

        return glosses

    def get_sign_video(self, gloss: str):
        """
        Returns path to the sign video if available.
        For now we will use one example video per class from the original data.
        """
        # You can later put representative videos in data/signs/
        possible = list(Path("data/raw").rglob(f"*{gloss}*"))
        # Fallback: just return None for now
        return None