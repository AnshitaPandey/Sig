"""
Gloss sequence → Natural English sentence using local Ollama (Phi / Phi-3)
"""

import requests
import json
from typing import List, Optional

class GlossToSentence:
    def __init__(
        self,
        model: str = "phi3",          # or "phi3:mini", "phi"
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout: float = 12.0,
    ):
        self.model = model
        self.url = f"{host}/api/generate"
        self.temperature = temperature
        self.timeout = timeout

    def _build_prompt(self, glosses: List[str]) -> str:
        gloss_str = " ".join(glosses)
        return f"""You are an expert Indian Sign Language (ISL) interpreter.
Convert the following sequence of ISL glosses into one natural, fluent English sentence.
- Keep the original meaning accurate.
- Add necessary grammar, articles, prepositions and punctuation.
- Do NOT invent extra information that is not in the glosses.
- Output ONLY the final English sentence. Nothing else.

Glosses: {gloss_str}

English sentence:"""

    def translate(self, glosses: List[str]) -> str:
        """
        Args:
            glosses: e.g. ["HELLO", "HOW_ARE_YOU", "FINE"]
        Returns:
            Natural English sentence
        """
        if not glosses:
            return ""

        # Simple single-gloss shortcuts (fast path)
        if len(glosses) == 1:
            g = glosses[0].upper()
            simple = {
                "HELLO": "Hello.",
                "THANK_YOU": "Thank you.",
                "YES": "Yes.",
                "NO": "No.",
                "PLEASE": "Please.",
                "SORRY": "Sorry.",
                "GOOD_MORNING": "Good morning.",
                "GOOD_AFTERNOON": "Good afternoon.",
                "GOOD_EVENING": "Good evening.",
                "GOOD_NIGHT": "Good night.",
            }
            if g in simple:
                return simple[g]

        prompt = self._build_prompt(glosses)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 60,          # keep answers short
            },
        }

        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            sentence = data.get("response", "").strip()

            # Clean up possible extra text
            sentence = sentence.split("\n")[0].strip()
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1]
            if not sentence.endswith((".", "?", "!")):
                sentence += "."

            return sentence

        except Exception as e:
            print(f"[LLM] Error: {e}")
            # Fallback: simple join
            return " ".join(g.replace("_", " ").title() for g in glosses) + "."

# Quick test
if __name__ == "__main__":
    translator = GlossToSentence(model="phi3")
    tests = [
        ["HELLO"],
        ["HELLO", "HOW_ARE_YOU"],
        ["GOOD_MORNING", "HOW_ARE_YOU"],
        ["I", "WANT", "WATER"],
        ["THANK_YOU", "VERY_MUCH"],
    ]
    for g in tests:
        print(f"{g}  →  {translator.translate(g)}")