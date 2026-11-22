# HealthTech Python Integration Changes

- `server.py`: added FastAPI WebSocket endpoint at `/ws` on port 8000.
- Reused the existing YOLO and MediaPipe implementation in `cv.py` without rewriting detection logic.
- YOLO is loaded once at server startup and reused across frames.
- OpenCV decodes browser JPEG frames and encodes annotated frames back as base64 JPEG.
- Results include `frame_id`, detections, bounding boxes, confidence, session state, therapy time, and annotation.
- Added `threading.Lock` protection around shared model use and `asyncio.to_thread` inference execution.
- Added reconnect-safe WebSocket behavior and invalid-frame/error handling.
- Made `YOLO_WEIGHTS` and `LANDMARKER_PATH` environment-overridable and project-relative by default.
- Added FastAPI/Uvicorn dependencies.
- Added `start_server.ps1` and `.env.example`.

## Files intentionally not ignored

- `models/best (1).pt`
- `face_landmarker.task`
