"""
Combined Bidirectional ISL Demo
- Text → ISL (Avatar)
- Voice → ISL (Avatar)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import speech_recognition as sr

from src.text_to_isl.sign_mapper import SignMapper
from src.text_to_isl.skeleton_avatar import SkeletonAvatar


class ISLDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bidirectional ISL Translation System")
        self.root.geometry("700x520")
        self.root.configure(bg="#1e1e2e")

        self.mapper = SignMapper()
        self.avatar = SkeletonAvatar()
        self.recognizer = sr.Recognizer()

        self._build_ui()

    def _build_ui(self):
        # Title
        title = tk.Label(
            self.root, text="Bidirectional ISL Translation System",
            font=("Segoe UI", 18, "bold"), fg="#89b4fa", bg="#1e1e2e"
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root, text="Text / Voice  →  Skeleton Avatar",
            font=("Segoe UI", 11), fg="#a6adc8", bg="#1e1e2e"
        )
        subtitle.pack(pady=(0, 20))

        # Input Frame
        input_frame = tk.Frame(self.root, bg="#1e1e2e")
        input_frame.pack(pady=10)

        self.entry = tk.Entry(
            input_frame, font=("Segoe UI", 13), width=40,
            bg="#313244", fg="white", insertbackground="white",
            relief="flat"
        )
        self.entry.pack(side=tk.LEFT, padx=(0, 10), ipady=8)
        self.entry.bind("<Return>", lambda e: self.process_text())

        btn_text = tk.Button(
            input_frame, text="Sign Text", font=("Segoe UI", 11, "bold"),
            bg="#89b4fa", fg="#1e1e2e", relief="flat", padx=15, pady=6,
            command=self.process_text
        )
        btn_text.pack(side=tk.LEFT, padx=5)

        btn_voice = tk.Button(
            input_frame, text="🎤 Speak", font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1", fg="#1e1e2e", relief="flat", padx=15, pady=6,
            command=self.process_voice
        )
        btn_voice.pack(side=tk.LEFT, padx=5)

        # Status
        self.status = tk.Label(
            self.root, text="Ready. Type a sentence or click Speak.",
            font=("Segoe UI", 10), fg="#a6adc8", bg="#1e1e2e"
        )
        self.status.pack(pady=15)

        # Gloss display
        self.gloss_label = tk.Label(
            self.root, text="", font=("Segoe UI", 12, "bold"),
            fg="#f9e2af", bg="#1e1e2e", wraplength=600
        )
        self.gloss_label.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Examples:  hello   |   how are you   |   thank you   |   good morning\n"
                 "The avatar window will open and perform the signs.",
            font=("Segoe UI", 9), fg="#6c7086", bg="#1e1e2e", justify="center"
        )
        instructions.pack(side=tk.BOTTOM, pady=20)

    def process_text(self):
        text = self.entry.get().strip()
        if not text:
            return
        self._run_signing(text)

    def process_voice(self):
        self.status.config(text="Listening... Speak now")
        self.root.update()

        def listen():
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                self.root.after(0, lambda: self._on_voice_result(text))
            except Exception as e:
                self.root.after(0, lambda: self.status.config(text=f"Error: {str(e)}"))

        threading.Thread(target=listen, daemon=True).start()

    def _on_voice_result(self, text):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.status.config(text=f"Recognized: {text}")
        self._run_signing(text)

    def _run_signing(self, text):
        glosses = self.mapper.text_to_glosses(text)

        if not glosses:
            self.gloss_label.config(text="Sorry, I don't know how to sign that yet.")
            self.status.config(text="Ready")
            return

        gloss_str = "  →  ".join(glosses)
        self.gloss_label.config(text=f"Glosses:  {gloss_str}")
        self.status.config(text="Avatar is signing...")
        self.root.update()

        # Run avatar in a separate thread so UI doesn't freeze
        def play():
            self.avatar.play_sequence(glosses)
            self.root.after(0, lambda: self.status.config(text="Ready"))

        threading.Thread(target=play, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ISLDemoApp(root)
    root.mainloop()