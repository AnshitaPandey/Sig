"""
Asynchronous Text-to-Speech Engine.
Camera and model inference continue while speech is playing.
"""

import queue
import threading
import time
from typing import Optional

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class TTSEngine:
    """
    Non-blocking TTS using a producer-consumer queue.
    """

    def __init__(
        self,
        rate: int = 160,
        volume: float = 0.9,
        cooldown_seconds: float = 1.8,
        enabled: bool = True,
    ):
        self.enabled = enabled and HAS_PYTTSX3
        self.cooldown = cooldown_seconds
        self.last_spoken_time = 0.0
        self.last_text = None

        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker = None

        if self.enabled:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()
        else:
            print("[TTS] pyttsx3 not available or disabled – TTS will be silent.")

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.3)
                if text is None:
                    break
                self.engine.say(text)
                self.engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def speak(self, text: str, force: bool = False):
        """Enqueue text for speaking. Suppresses rapid duplicates."""
        if not self.enabled or not text:
            return

        now = time.time()
        if not force:
            if text == self.last_text and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < 0.6:
                return

        self.last_text = text
        self.last_spoken_time = now
        self.queue.put(text)

    def stop(self):
        self._stop_event.set()
        self.queue.put(None)
        if self.worker:
            self.worker.join(timeout=2.0)
