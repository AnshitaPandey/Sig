import cv2
from pathlib import Path
import time

class SignVideoPlayer:
    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.gloss_to_videos = self._build_index()

    def _build_index(self):
        """
        Creates a mapping from gloss name to list of video paths.
        Example: "HELLO" → [path1, path2, ...]
        """
        index = {}
        
        # Animals
        animals_dir = self.raw_data_dir / "Animals"
        if animals_dir.exists():
            for folder in animals_dir.iterdir():
                if folder.is_dir():
                    gloss = self._clean_name(folder.name)
                    videos = list(folder.glob("*.MOV")) + list(folder.glob("*.MP4")) + list(folder.glob("*.mp4"))
                    # Also check Extra subfolder
                    extra = folder / "Extra"
                    if extra.exists():
                        videos += list(extra.glob("*.MOV")) + list(extra.glob("*.MP4"))
                    if videos:
                        index[gloss] = videos

        # Greetings
        greetings_dir = self.raw_data_dir / "Greetings"
        if greetings_dir.exists():
            for folder in greetings_dir.iterdir():
                if folder.is_dir():
                    gloss = self._clean_name(folder.name)
                    videos = list(folder.glob("*.MOV")) + list(folder.glob("*.MP4")) + list(folder.glob("*.mp4"))
                    if videos:
                        index[gloss] = videos

        print(f"Indexed {len(index)} glosses with videos.")
        return index

    def _clean_name(self, folder_name: str) -> str:
        """Convert '1. Dog' or '48. Hello' → 'DOG' / 'HELLO' """
        import re
        name = re.sub(r'^\d+\.\s*', '', folder_name)
        name = name.upper().replace(' ', '_')
        return name

    def play(self, gloss: str, max_duration=4.0):
        """
        Plays one video for the given gloss.
        """
        videos = self.gloss_to_videos.get(gloss.upper())
        if not videos:
            print(f"No video found for gloss: {gloss}")
            return False

        video_path = str(videos[0])  # take the first available video
        print(f"Playing: {gloss} → {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False

        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for nicer display
            frame = cv2.resize(frame, (640, 480))
            cv2.putText(frame, f"ISL: {gloss}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("ISL Sign Output", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            if time.time() - start_time > max_duration:
                break

        cap.release()
        cv2.destroyWindow("ISL Sign Output")
        return True

    def play_sequence(self, glosses: list):
        """Play multiple glosses one after another"""
        for gloss in glosses:
            self.play(gloss)
            time.sleep(0.4)