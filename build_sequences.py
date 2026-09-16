import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def main():
    landmarks_dir = Path("data/processed/landmarks")
    output_dir = Path("data/processed")
    seq_len = 45          # same as config
    min_frames = 20

    # Load label map
    with open(output_dir / "label_map.json") as f:
        label_map = json.load(f)

    inv_label_map = {v: k for k, v in label_map.items()}
    print("Classes:", label_map)

    X, y = [], []

    for class_name, class_idx in label_map.items():
        class_folder = landmarks_dir / class_name
        if not class_folder.exists():
            print(f"Warning: {class_name} folder not found")
            continue

        npy_files = list(class_folder.glob("*.npy"))
        print(f"{class_name}: {len(npy_files)} videos")

        for npy_path in npy_files:
            lm = np.load(npy_path)  # (T, 75, 4)

            if len(lm) < min_frames:
                continue

            # Simple sampling to fixed length
            if len(lm) >= seq_len:
                # take middle portion
                start = (len(lm) - seq_len) // 2
                seq = lm[start:start + seq_len]
            else:
                # pad by repeating last frame
                pad = np.repeat(lm[-1:], seq_len - len(lm), axis=0)
                seq = np.concatenate([lm, pad], axis=0)

            # Flatten to (T, 300)
            seq = seq.reshape(seq_len, -1).astype(np.float32)
            X.append(seq)
            y.append(class_idx)

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    print(f"\nTotal sequences: {len(y)}")
    print(f"Shape: {X.shape}")

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    np.savez_compressed(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(output_dir / "test.npz", X=X_test, y=y_test)

    print(f"Train: {len(y_train)}")
    print(f"Val  : {len(y_val)}")
    print(f"Test : {len(y_test)}")
    print("Saved train/val/test .npz files")

if __name__ == "__main__":
    main()