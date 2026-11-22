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
python -m pip install -r requirements.txt
```

You also need:

- A YOLO weights file containing `eye_normal` and `eye_patched` classes
- A MediaPipe `face_landmarker.task` model

The repository includes both expected model files:

- `models/best (1).pt`
- `face_landmarker.task`

Keep both files in the project root as shown. They are not generated at runtime.

## Configuration

The server uses the included weights file by default:

```python
YOLO_WEIGHTS = "models/best (1).pt"
```

Override `YOLO_WEIGHTS` or `LANDMARKER_PATH` with environment variables when using a different location. The MediaPipe landmark file is optional for the WebSocket server; without it, YOLO detections and annotated frames still work, but blink/eye-closure analysis is unavailable.

## Run the Python WebSocket server

```powershell
cd /path/to/HealthTech
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Or run the included script:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_server.ps1
```

The server accepts browser frames at `ws://127.0.0.1:8000/ws` and returns the frame ID, annotated JPEG, detections, and session state. It does not open a camera; the React application owns the camera stream.

## Run the React application

The React application is in the separate PATCHY repository. After cloning it:

```powershell
cd /path/to/Patchy.Api/Frontend
npm install
npm run dev
```

Vite keeps `/api` pointed at the .NET API and proxies `/ws-python` to the Python `/ws` endpoint. Set `PYTHON_URL` if the Python server uses a different host or port. Set `VITE_YOLO_WS_URL` to an absolute `ws://` or `wss://` URL when the frontend is not using the Vite proxy.

The original standalone monitor remains available with `python cv.py`; it still owns its own camera only when run directly.
