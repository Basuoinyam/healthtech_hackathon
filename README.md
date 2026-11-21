# HealthTech Lazy-Eye Monitor

A webcam-based patch-session monitor for amblyopia (lazy-eye) therapy. It combines a YOLO eye-patch detector with MediaPipe Face Landmarker to track session states and alert when the uncovered eye closes.

## Features

- Detects normal and patched eyes with YOLO
- Tracks therapy, both-open, both-patched, and lost-detection states
- Measures therapy and paused time
- Writes session events and totals to `session_log.json`
- Optionally shows an annotated video window

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLO
- MediaPipe

Install the Python dependencies:

```powershell
python -m pip install opencv-python numpy ultralytics mediapipe
```

You also need:

- A YOLO weights file containing `eye_normal` and `eye_patched` classes
- A MediaPipe `face_landmarker.task` model

## Configuration

Update the paths and runtime settings near the top of `cv.py`:

```python
YOLO_WEIGHTS = r"D:\Lazy_eye\models\best (1).pt"
VIDEO_SOURCE = 0
LANDMARKER_PATH = r"D:\Lazy_eye\face_landmarker.task"
```

Set `VIDEO_SOURCE` to a camera index or a video file path. Set `OUTPUT_VIDEO` to a filename to save the annotated feed, or leave it as `None`.

## Run

```powershell
python cv.py
```

Press `Q` or `Esc` to stop the session. The program saves the final therapy totals and state events to `session_log.json`.
