# Bidirectional Indian Sign Language Translation System

**B.Tech. Project, Department of Electrical Engineering, IIT Indore**
Author: Anshita Pandey (Roll No. 230002010) | Supervisor: Prof. Vivek Kanhangad

A real-time, two-way communication system for Indian Sign Language (ISL). It recognises ISL signs from a live webcam stream and speaks them aloud, and it converts typed or spoken text back into ISL glosses shown on a skeleton or 3D avatar. Both directions share one 263-gloss vocabulary taken from the INCLUDE dataset.

| Direction | What it does |
|---|---|
| **ISL → Text / Speech** | Webcam → MediaPipe landmarks → Transformer classifier → stabilised caption → offline speech |
| **Text / Speech → ISL** | Typed or spoken input → gloss mapping → recorded landmark playback on a skeleton avatar (VRM 3D avatar partially integrated) |

---

## Highlights

- **90.66 % test accuracy** and **90.75 % macro F1** on the full INCLUDE vocabulary (263 classes, 1285 held-out test sequences) with a 1.06M-parameter landmark-based Transformer encoder.
- **Landmark-based input**: 75 MediaPipe Holistic points × 4 values = 300 features per frame, 45-frame windows. Runs on a laptop CPU, ignores background and clothing, and never needs raw video to be stored.
- **Five model families compared**: GRU, LSTM, TCN, hybrid TCN–GRU and Transformer encoder.
- **Stable live captions**: confidence gating, majority voting and cooldown suppress flicker; text-to-speech runs in a separate thread so the camera never blocks.
- **Mostly offline**: recognition, speech output and avatar playback work without internet. Only the speech-input path (Google Web Speech API) needs a connection.
- **Companion app** ("ISL Connect"): a floating, always-on-top Tkinter window designed to sit over call or chat windows.

---

## Results

Final model: Transformer encoder, evaluated offline on the held-out INCLUDE test split.

| Metric | Value |
|---|---|
| Test accuracy | 90.66 % (1165 / 1285) |
| Macro precision | 92.26 % |
| Macro recall | 91.12 % |
| Macro F1 | 90.75 % |
| Best validation F1 | ~91.55 % |

Macro scores are computed over the 262 classes that occur in the test set (see *Known issues* for the spurious `MEANS_OF_TRANSPORTATION` label).

| Stage | Classes | Model | Test accuracy |
|---|---|---|---|
| Mid-semester | 17 | Causal TCN | 66.7 % |
| Final | 263 | Transformer encoder | 90.66 % |

Per-class results (3–8 test sequences per class, so noisy): 144 classes reach F1 = 1.0 and 178 reach F1 ≥ 0.9. Weakest classes include MINUTE, RED, PAGE and GIRL. The most common confusions are between related signs (GIRL/BOY, DAUGHTER/SON, WIFE/HUSBAND, QUEEN/KING, SHE/HE, I/YOU), because the landmark features carry no finger-level appearance and no facial cues.

> **Important:** these are offline numbers on pre-recorded INCLUDE videos. Live webcam accuracy is lower because of a domain gap (camera angle, distance, lighting, signer, timing). See *Known issues*.

---

## System overview

```
ISL → Text / Speech
Webcam frame
   ↓  MediaPipe Holistic (pose + both hands)
75 landmarks × 4 values = 300 features / frame
   ↓  buffer of the latest 45 frames
Temporal Transformer (263 classes)
   ↓  softmax + confidence threshold
Stabilisation (majority vote + cooldown)
   ↓
Caption on screen  +  asynchronous offline TTS (pyttsx3)

Text / Speech → ISL
Typed text   or   speech → text (Google Web Speech API)
   ↓  SignMapper (rule-based phrase / word → gloss)
Gloss list
   ↓  recorded landmark sequence per gloss
Skeleton avatar (25 fps)   |   VRM 3D avatar in browser (partial)
```

### Model

| Item | Setting |
|---|---|
| Architecture | Transformer encoder, Pre-LayerNorm, GELU, no causal mask |
| Size | d_model 160, 5 heads, 4 layers, feed-forward 320, ~1.06M parameters |
| Head | last-frame embedding + mean-pooled embedding → MLP → 263 scores |
| Loss | Cross-entropy with class weights w_k = N / (K · n_k) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule |
| Regularisation | Dropout 0.2, gradient clipping at 1.0 |
| Augmentation | Temporal cropping, time-warping, additive noise, feature-scale perturbation |
| Selection | Best validation macro F1; early stopping, patience 10, max 100 epochs |
| Seed | 42 |

### Dataset

[INCLUDE](https://github.com/AI4Bharat/INCLUDE) (AI4Bharat): isolated ISL signs in 15 categories (Adjectives, Animals, Clothes, Colours, Days and Time, Electronics, Greetings, Home, Jobs, Means of Transportation, People, Places, Pronouns, Seasons, Society).

| Item | Value |
|---|---|
| Classes | 263 |
| Sequences after filtering | 8566 |
| Train / val / test | 6188 / 1093 / 1285 (stratified, seed 42) |
| Sequence format | `(N, 45, 300)` in `.npz` + `label_map.json` |

Preprocessing: videos under 15 frames are dropped; longer videos are uniformly sub-sampled to 45 frames and shorter ones padded by repeating the last frame; missing detections are zero-filled; hand landmarks get visibility 1. MediaPipe Holistic runs with model complexity 1 and detection/tracking confidence 0.5.

---

## Repository structure

```
Sig/
├── configs/
│   └── default.yaml              # hyperparameters
├── results/                      # test report, confusion matrix
├── src/
│   ├── data/                     # dataset loading, normalisation, synthetic data
│   ├── models/                   # temporal model definitions
│   ├── training/train.py         # training and evaluation loop
│   ├── inference/                # predictor, smoothing, tts, llm_translator
│   ├── text_to_isl/              # sign_mapper, skeleton_avatar, video_player
│   └── utils/                    # config, logger
├── extract_full_dataset.py       # Holistic landmark extraction (full INCLUDE)
├── extract_real_data.py          # early extraction script for real videos
├── build_sequences.py            # 45-frame sequences + stratified split
├── train.py                      # training entry point
├── train_hybrid_tcn_gru.py       # early stand-alone hybrid TCN–GRU experiment
├── compare_models.py             # model comparison
├── evaluate_only.py              # offline test evaluation from a checkpoint
├── run_realtime.py               # live recognition with speech
├── run_realtime_sentence.py      # live recognition, sentence-level output
├── companion_app.py              # "ISL Connect" Tkinter app
├── run_text_to_isl.py            # text → ISL
├── run_voice_to_isl.py           # voice → ISL
├── run_bidirectional.py          # both directions
├── run_combined_demo.py          # combined demo
├── test_avatar.py                # avatar playback test
└── requirements.txt
```

`data/` and `models/` are created locally by the steps below and are not stored in the repository.

---

## Getting started

### Requirements

- Python 3.9+ (3.10 recommended)
- A webcam for live demos
- No GPU needed; training and inference were done mainly on CPU
- Optional: [Ollama](https://ollama.com) with the Phi-3 model for the gloss-to-sentence module

Main libraries: PyTorch, MediaPipe, OpenCV, NumPy, `pyttsx3` (offline TTS), `SpeechRecognition` (speech input), Tkinter (companion app).

### Install

```bash
git clone https://github.com/AnshitaPandey/Sig.git
cd Sig
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Get the dataset

Download INCLUDE from its authors and place the videos under `data/raw/`, keeping the category folders.

### 2. Extract landmarks and build sequences

```bash
python extract_full_dataset.py
python build_sequences.py
```

This produces `train.npz`, `val.npz`, `test.npz` and `label_map.json`. The same `label_map.json` must be used for evaluation and inference so class indices stay consistent in both directions.

### 3. Train

```bash
python train.py --model tcn --epochs 80       # see: python train.py --help
```

Hyperparameters live in `configs/default.yaml`. The best checkpoint (by validation macro F1) is saved under `models/checkpoints/`.

### 4. Evaluate

```bash
python evaluate_only.py
```

Regenerates accuracy, macro precision/recall/F1, the per-class report and the confusion matrix from the saved checkpoint.

### 5. Run the demos

```bash
python run_realtime.py             # live ISL → text + speech (press q to quit)
python run_realtime_sentence.py    # groups glosses into a sentence after a pause
python companion_app.py            # floating ISL Connect window
python run_text_to_isl.py          # typed text → ISL avatar
python run_voice_to_isl.py         # speech → ISL avatar (needs internet for STT)
python run_bidirectional.py        # both directions
```

---

## Real-time parameters

| Parameter | Real-time predictor | Companion app |
|---|---|---|
| Window length | 45 frames | 45 frames |
| Inference stride | every 3 frames | every frame once buffer is full |
| Confidence threshold | 0.65 (0.40 for no-sign) | 0.52 |
| Majority vote | 4 of last 7 | 2 of last 3 |
| Repeat suppression | cooldown of 15 inferences; TTS cooldown 1.8 s | show gloss only when it changes |
| MediaPipe complexity | 1 | 0 (faster) |

In the sentence-level demo a 1.6 s pause ends a gloss group, which a template builder turns into a sentence (HELLO + HOW_ARE_YOU → "Hello, how are you?").

---

## Text / speech to ISL

- **SignMapper** matches multi-word phrases first, then single words, against a hand-written dictionary of 21 English words and phrases mapped to 17 glosses. No ISL grammar reordering and no fingerspelling fallback.
- **Skeleton avatar** draws one stored MediaPipe landmark sequence per gloss at 25 fps.
- **Speech input** uses the Google Web Speech API (5 s listen timeout, 4 s phrase limit); speech output uses `pyttsx3`.
- **LLM module** (standalone): sends a gloss sequence to a local Ollama model (Phi-3, temperature 0.3, 60 tokens, 12 s timeout) to get one fluent English sentence. Common single-gloss greetings are answered without the model, and glosses are simply joined if the model is unavailable. Not yet connected to the companion app.
- **VRM 3D avatar** (browser, Three.js + three-vrm, fed by a Python WebSocket server): model loads and landmark frames stream; Kalidokit → VRM bone animation is still pending.

---

## Known issues and limitations

- **Domain gap**: offline accuracy (~90 %) does not fully carry over to the live webcam. Things still to check: webcam mirroring (INCLUDE videos are not mirrored, so left and right hands swap), MediaPipe complexity 0 live versus 1 during extraction, identical normalisation in training and inference, and the short voting window.
- **Isolated signs only**: no continuous signing, no sentence-level ISL grammar, no face or non-manual signals.
- **Spurious class**: `MEANS_OF_TRANSPORTATION` is a category folder that was treated as a class and has no test samples.
- **Small reverse vocabulary**: SignMapper covers 17 of 263 glosses.
- **Avatar**: VRM bone mapping incomplete; transitions between signs are not smoothed.
- **Speech input** requires an internet connection.

---

## Roadmap

1. Webcam domain adaptation and personal fine-tuning; align live preprocessing with training.
2. Remove the spurious class from the label set.
3. Expand SignMapper to all glosses, add ISL grammar reordering and fingerspelling fallback; connect the LLM module to the companion app.
4. Complete Kalidokit → VRM full-body and hand animation.
5. Middleware packaging and a single-platform integration prototype (e.g. a video-calling app); quantitative live latency and robustness evaluation.

---

## Privacy

Only landmark coordinates are used by the model, so background, clothing and the signer's appearance are discarded. Recognition and speech output run locally.

---

## Acknowledgements

- [INCLUDE](https://github.com/AI4Bharat/INCLUDE) dataset (AI4Bharat): A. Sridhar et al., *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe), PyTorch, OpenCV, Three.js and three-vrm communities.
- Prof. Vivek Kanhangad, Department of Electrical Engineering, IIT Indore, for guidance.

## Citation

```
A. Pandey, "Bidirectional Indian Sign Language Translation System for Accessible
Digital Communication," B.Tech. Project, Dept. of Electrical Engineering,
IIT Indore, 2026. https://github.com/AnshitaPandey/Sig
```

## License

No license has been added yet. Add a `LICENSE` file (for example MIT) if you want others to reuse the code.
