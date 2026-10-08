"""
Extract MediaPipe Holistic landmarks from the full INCLUDE-50 dataset.
Handles category folders and numbered class names.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import re
from tqdm import tqdm
import time

# ====================== CONFIG ======================
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/landmarks")
MAX_FRAMES = 45          # fixed sequence length
MIN_FRAMES = 15          # ignore very short videos

# Categories to process (skip _test)
CATEGORIES = [
    "Adjectives", "Animals", "Clothes", "Colours", "Days_and_Time",
    "Electronics", "Greetings", "Home", "Jobs", "Means_of_Transportation",
    "People", "Places", "Pronouns", "Seasons", "Society"
]

def clean_name(folder_name: str) -> str:
    """Convert '1. Dog' or '48. Hello' → 'DOG' / 'HELLO' """
    name = re.sub(r'^\d+\.\s*', '', folder_name)   # remove leading number
    name = name.strip().upper().replace(" ", "_")
    name = re.sub(r'[^A-Z0-9_]', '', name)        # keep only safe characters
    return name

def extract_landmarks_from_video(video_path: Path, holistic) -> np.ndarray | None:
    """Extract landmarks and return array of shape (T, 75, 4)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        lm = np.zeros((75, 4), dtype=np.float32)

        if results.pose_landmarks:
            for i, p in enumerate(results.pose_landmarks.landmark):
                lm[i] = [p.x, p.y, p.z, p.visibility]

        if results.left_hand_landmarks:
            for i, p in enumerate(results.left_hand_landmarks.landmark):
                lm[33 + i] = [p.x, p.y, p.z, 1.0]

        if results.right_hand_landmarks:
            for i, p in enumerate(results.right_hand_landmarks.landmark):
                lm[54 + i] = [p.x, p.y, p.z, 1.0]

        frames.append(lm)

    cap.release()

    if len(frames) < MIN_FRAMES:
        return None

    # Pad or truncate to MAX_FRAMES
    frames = np.array(frames)
    if len(frames) > MAX_FRAMES:
        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
        frames = frames[indices]
    else:
        # Pad by repeating last frame
        pad = np.repeat(frames[-1:], MAX_FRAMES - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    return frames  # (45, 75, 4)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    total_videos = 0
    success = 0
    skipped = 0

    print("=" * 60)
    print("Starting full INCLUDE-50 landmark extraction...")
    print("=" * 60)

    for category in CATEGORIES:
        cat_path = RAW_DIR / category
        if not cat_path.exists():
            print(f"Category not found: {category}")
            continue

        print(f"\n>>> Processing category: {category}")

        class_folders = [d for d in cat_path.iterdir() if d.is_dir()]
        for class_dir in tqdm(class_folders, desc=category):
            gloss = clean_name(class_dir.name)
            if not gloss:
                continue

            out_class_dir = OUTPUT_DIR / gloss
            out_class_dir.mkdir(exist_ok=True)

            videos = list(class_dir.glob("*.MOV")) + list(class_dir.glob("*.mp4")) + \
                     list(class_dir.glob("*.MP4")) + list(class_dir.glob("*.mov"))

            # Also check Extra subfolder if exists
            extra = class_dir / "Extra"
            if extra.exists():
                videos += list(extra.glob("*.MOV")) + list(extra.glob("*.mp4")) + \
                          list(extra.glob("*.MP4")) + list(extra.glob("*.mov"))

            for i, video in enumerate(videos):
                total_videos += 1
                out_file = out_class_dir / f"{gloss}_{i:03d}.npy"

                # Skip if already extracted
                if out_file.exists():
                    success += 1
                    continue

                try:
                    landmarks = extract_landmarks_from_video(video, holistic)
                    if landmarks is not None:
                        np.save(out_file, landmarks)
                        success += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Error on {video.name}: {e}")
                    skipped += 1

    holistic.close()

    print("\n" + "=" * 60)
    print("Extraction finished!")
    print(f"Total videos processed : {total_videos}")
    print(f"Successfully extracted : {success}")
    print(f"Skipped / failed       : {skipped}")
    print(f"Landmarks saved in     : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()