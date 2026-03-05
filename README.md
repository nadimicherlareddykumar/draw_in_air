# ✋ AI Hand Gesture Controlled Virtual Whiteboard (draw_in_air)

A real-time computer vision project that lets you **draw in the air** using only your hands — no mouse, no touch screen required.

Built with **Python · OpenCV · MediaPipe · NumPy · scikit-learn**.

---

## 📸 What it does

| Feature | Detail |
|---|---|
| Real-time hand tracking | MediaPipe 21-landmark skeleton on up to 2 hands |
| Freehand drawing | Smooth air-drawing with your index finger |
| Shape drawing | Lines, rectangles, circles — gesture-controlled |
| Colour palette | 8 colours including an eraser, always visible in toolbar |
| Two-hand support | Right = draw, Left = pick colours from toolbar |
| Gesture recognition | Rule-based + optional ML (Random Forest / SVM) |
| Undo / Redo | 20-step history stack |
| Save drawing | PNG with timestamp, auto-saved to `saved_drawings/` |
| Clear board | Gesture or keyboard |
| ML data collection | Built-in record mode — no external tools needed |

---

## 🗂️ Project Structure

```
draw_in_air/
│
├── main.py                # 🚀 Entry point – run this
├── hand_tracker.py        # MediaPipe wrapper + finger-state helpers
├── gesture_classifier.py  # ML data collection, training, prediction
├── utils.py               # Toolbar, canvas ops, undo/redo, HUD
│
├── dataset/               # Auto-created; gesture_data.csv goes here
├── models/                # Auto-created; gesture_model.pkl saved here
├── saved_drawings/        # Auto-created; PNGs saved here
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1 — Clone / download the project

```bash
git clone https://github.com/nadimicherlareddykumar/draw_in_air.git
cd draw_in_air
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9 – 3.11** recommended (MediaPipe 0.10 requires ≤ 3.11 on some platforms).

---

## ▶️ Running the app

```bash
python main.py
```

The webcam window opens immediately. Press **H** at any time for the in-app help overlay.

> If your webcam isn't detected, change `CAMERA_ID = 0` at the top of `main.py` to `1` or `2`.

---

## 🖐️ Gesture Reference

### Right hand (drawing hand)

| Fingers raised | Gesture | Action |
|---|---|---|
| Index only | `draw` | Draw / interact with toolbar |
| Index + Middle | `erase` / select | Lift pen; hover toolbar to pick colour |
| Index + Middle + Ring | `change_color` | Cycle through shape modes |
| Index + Middle + Ring + Pinky | `save` | Save drawing as PNG |
| All 5 fingers | `clear` | Clear the entire board |
| Fist (none) | `idle` | Do nothing |

### Left hand

Hover left index finger over the colour swatches at the top to instantly switch colour without interrupting your right-hand drawing.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `Z` | Undo |
| `Y` / `Shift+Z` | Redo |
| `S` | Save drawing |
| `C` | Clear board |
| `1` | Freehand mode |
| `2` | Line mode |
| `3` | Rectangle mode |
| `4` | Circle mode |
| `T` | Train ML model |
| `M` | Toggle ML data-collection mode |
| `0` – `4` | Choose collection label (in collect mode) |
| `H` | Help |
| `Q` / `Esc` | Quit |

---

## 🤖 ML Gesture Classification (optional)

The app works perfectly with **rule-based gesture detection** out of the box. If you want the ML layer:

### Step 1 — Collect samples

1. Launch the app: `python main.py`
2. Press `M` to enter **collect mode** (green banner appears)
3. Press `0`–`4` to choose which gesture label to record:
   - `0` = draw | `1` = erase | `2` = change_color | `3` = save | `4` = clear
4. Hold the gesture in front of the camera — samples are recorded automatically (~10/sec)
5. Collect **≥ 20 samples per class** (more is better; 100+ per class is ideal)
6. Press `M` again to stop collecting

### Step 2 — Train

```
Press  T  in the app window
```

or run from the terminal:

```python
from gesture_classifier import GestureClassifier
gc = GestureClassifier()
gc.train(verbose=True)
```

The trained model is saved to `models/gesture_model.pkl` and loaded automatically on next launch.

### Step 3 — Predict

The model activates automatically once `gesture_model.pkl` exists. Predictions with < 70 % confidence fall back to rule-based detection.

---

## 🎨 Colour Palette

| Swatch | Colour |
|---|---|
| 🔵 | Blue |
| 🟢 | Green |
| 🔴 | Red |
| 🟡 | Yellow |
| 🟣 | Purple |
| 🟠 | Orange |
| ⬜ | White |
| ⬛ | Eraser (black) |

---

## 🛠️ Configuration (top of `main.py`)

```python
CAMERA_ID        = 0      # Webcam index
FRAME_WIDTH      = 1280   # Resolution width
FRAME_HEIGHT     = 720    # Resolution height
BRUSH_THICKNESS  = 5      # Drawing line width (px)
ERASER_THICKNESS = 40     # Eraser size (px)
SMOOTH_ALPHA     = 0.45   # Finger-position smoothing (0=none, 1=frozen)
ML_CONFIDENCE_THRESHOLD = 0.70  # Min ML confidence to override rule-based
```

---

## 🔮 Possible Extensions

- **Voice commands** — use `SpeechRecognition` to switch modes by speaking
- **Streamlit web version** — serve the whiteboard through a browser
- **Multi-user collaboration** — share the canvas over WebSockets
- **Custom gestures** — add more labels to `GESTURE_LABELS` and retrain
- **Text recognition** — detect letters drawn on the canvas with Tesseract OCR
- **Export to SVG / PDF** — convert `canvas` numpy array via `svgwrite` or `reportlab`

---

## 📋 Requirements

```
opencv-python >= 4.8.0
mediapipe     >= 0.10.0
numpy         >= 1.24.0
scikit-learn  >= 1.3.0
```

---

## 📄 Licence

MIT — free to use, modify, and distribute.