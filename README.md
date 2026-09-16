# Bidirectional Indian Sign Language Translation System  
**Bachelor’s Thesis Project – Phase 1**

**Title:** Bidirectional Indian Sign Language Translation System for Accessible Digital Communication

---

## Phase 1 Scope (Completed)

This release implements the **core Deep Learning + Computer Vision** pipeline:

- Real-time **ISL → Text + Speech**
- MediaPipe Holistic landmark extraction (75 landmarks × 4)
- Temporal models: **GRU (baseline)** + **TCN (recommended)** + **Hybrid TCN-GRU**
- Sliding-window streaming inference
- Prediction stabilization (confidence + majority vote + cooldown)
- Asynchronous offline TTS
- Modular, reproducible codebase

**Phase 2** (planned) will add:
- Text → ISL
- Voice → ISL (via STT)
- Full bidirectional web interface

---

## Project Structure

```
isl-translation/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── raw/                      # Original INCLUDE videos (user-provided)
│   ├── processed/                # .npz sequences + label_map.json
│   └── splits/
├── models/
│   └── checkpoints/
│       └── best_model.pth
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── normalize.py
│   │   └── synthetic.py          # Synthetic data for development
│   ├── models/
│   │   ├── gru.py                # Causal baseline
│   │   ├── tcn.py                # Recommended real-time model
│   │   └── hybrid.py
│   ├── training/
│   │   └── train.py
│   ├── inference/
│   │   ├── predictor.py          # Full real-time pipeline
│   │   ├── smoothing.py          # Stabilization logic
│   │   └── tts.py
│   └── utils/
├── train.py                      # Training entry point
├── run_realtime.py               # Webcam demo entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd isl-translation
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train a Model (Synthetic data for demo)

```bash
# Recommended model
python train.py --model tcn --epochs 25

# Baseline
python train.py --model gru --epochs 20

# Hybrid
python train.py --model hybrid --epochs 25
```

Training automatically generates a synthetic landmark dataset so you can verify the entire pipeline without downloading INCLUDE immediately.

### 3. Run Real-time Webcam Demo

```bash
python run_realtime.py
```

Press `q` to quit.

You should see:
- Live webcam with pose + hand landmarks
- Current prediction + confidence
- Inference latency
- Spoken output when a sign is stably recognized

---

## Using the Real INCLUDE Dataset

1. Download the INCLUDE / INCLUDE-50 dataset.
2. Place videos under `data/raw/`.
3. Implement / adapt the landmark extraction script (MediaPipe Holistic) to produce sequences of shape `(T, 75, 4)`.
4. Run the same normalization → sequence construction → save as `train.npz / val.npz / test.npz`.
5. **Important**: Create `label_map.json` **only from the training split** and reuse it for val/test/inference.

The rest of the pipeline remains unchanged.

---

## Model Architecture Decisions

| Model       | Streaming Suitability | Latency | Notes                          |
|-------------|-----------------------|---------|--------------------------------|
| GRU         | Excellent             | Low     | Strong academic baseline       |
| BiLSTM      | Poor                  | —       | Avoided for real-time reasons  |
| **TCN**     | Excellent             | Very Low| **Recommended final model**    |
| Hybrid      | Excellent             | Low     | Good alternative               |

The TCN uses dilated causal convolutions and is fully compatible with sliding-window inference.

---

## Real-time Pipeline

```
Webcam Frame
    ↓
MediaPipe Holistic (Pose + Hands)
    ↓
75 × 4 landmarks
    ↓
Normalization (identical to training)
    ↓
Temporal Ring Buffer
    ↓
Sliding Window (T=45, stride=3)
    ↓
TCN / GRU Model
    ↓
Softmax + Confidence Filter
    ↓
Stabilization (majority vote + debounce + cooldown)
    ↓
Text Display + Asynchronous TTS
```

---

## Key Engineering Rules Followed

- Fully causal models only (no BiLSTM in final real-time path)
- Train-only label mapping
- Identical normalization for train & inference
- Modular components
- Configurable everything via YAML
- No hard-coded predictions
- Proper early stopping + checkpointing
- Reproducible seeds

---

## Next Steps (Phase 2)

1. Integrate real INCLUDE data extraction
2. Add Text → ISL (retrieval-based first)
3. Add Speech-to-Text → ISL (Voice to Sign)
4. Minimal FastAPI + WebSocket interface
5. Full system benchmarking (FPS, latency, robustness)

---

## Citation / Thesis Notes

This Phase 1 deliverable focuses on:

- Streaming temporal modeling for Indian Sign Language landmarks
- Real-time prediction stabilization
- Clean separation of offline training vs online inference
- Practical performance on student hardware

These points form the core technical contribution of the BTP.

---

**Author**: BTP Student  
**Framework**: PyTorch + MediaPipe + OpenCV  
**Status**: Phase 1 Complete
