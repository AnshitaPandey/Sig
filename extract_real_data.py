import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import json

mp_holistic = mp.solutions.holistic

def clean_class_name(folder_name: str) -> str:
    """Convert '1. Dog' → 'DOG', '48. Hello' → 'HELLO' """
    name = re.sub(r'^\d+\.\s*', '', folder_name)  # remove number prefix
    name = name.upper().replace(' ', '_')
    return name

def extract_from_video(video_path: str, max_frames: int = 90) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    landmarks_list = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened() and len(landmarks_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frame_lm = np.zeros((75, 4), dtype=np.float32)

            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    frame_lm[i] = [lm.x, lm.y, lm.z, lm.visibility]

            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    frame_lm[33 + i] = [lm.x, lm.y, lm.z, 1.0]

            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    frame_lm[54 + i] = [lm.x, lm.y, lm.z, 1.0]

            landmarks_list.append(frame_lm)

    cap.release()
    if len(landmarks_list) < 15:  # too short
        return None
    return np.stack(landmarks_list)


def main():
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed/landmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all class folders
    class_folders = []
    for category in ["Animals", "Greetings"]:
        cat_path = raw_dir / category
        if cat_path.exists():
            for folder in cat_path.iterdir():
                if folder.is_dir() and not folder.name.startswith('_'):
                    class_folders.append(folder)

    print(f"Found {len(class_folders)} classes")

    label_map = {}
    class_idx = 0

    for folder in tqdm(class_folders, desc="Classes"):
        class_name = clean_class_name(folder.name)
        if class_name not in label_map:
            label_map[class_name] = class_idx
            class_idx += 1

        class_out = output_dir / class_name
        class_out.mkdir(exist_ok=True)

        videos = list(folder.glob("*.MOV")) + list(folder.glob("*.MP4")) + list(folder.glob("*.mp4"))
        # also check Extra subfolder
        extra = folder / "Extra"
        if extra.exists():
            videos += list(extra.glob("*.MOV")) + list(extra.glob("*.MP4"))

        for video in videos:
            out_path = class_out / (video.stem + ".npy")
            if out_path.exists():
                continue

            lm = extract_from_video(video)
            if lm is not None:
                np.save(out_path, lm)

    # Save label map
    with open("data/processed/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\nLabel map:")
    for k, v in label_map.items():
        print(f"  {v}: {k}")
    print(f"\nLandmarks saved to: {output_dir}")


if __name__ == "__main__":
    main()