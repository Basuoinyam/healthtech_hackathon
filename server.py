"""FastAPI WebSocket adapter for the existing cv.py detection pipeline."""

import asyncio
import base64
import binascii
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from cv import SessionLogic, analyze, create_landmarker, draw


MODEL_PATH = os.getenv("YOLO_WEIGHTS", os.path.join(os.path.dirname(__file__), "models", "best (1).pt"))
_model = None
_model_lock = threading.Lock()
_ids = None
_mp = None
_landmarker = None


def load_model():
    global _model, _ids
    if _model is not None:
        return
    from ultralytics import YOLO

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"YOLO weights not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    names = model.names
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    name_to_id = {str(name): int(class_id) for class_id, name in names.items()}
    missing = [name for name in ("eye_patched", "eye_normal") if name not in name_to_id]
    if missing:
        raise ValueError(f"YOLO model is missing classes: {missing}. Model classes: {list(name_to_id)}")
    with _model_lock:
        if _model is None:
            _model = model
            _ids = {"patched": name_to_id["eye_patched"], "normal": name_to_id["eye_normal"]}


def load_landmarker():
    global _mp, _landmarker
    path = os.getenv("LANDMARKER_PATH", os.path.join(os.path.dirname(__file__), "face_landmarker.task"))
    if os.path.isfile(path):
        _mp, _landmarker = create_landmarker(path)


def process_frame(frame: np.ndarray, logic: SessionLogic, timestamp_ms: int):
    with _model_lock:
        raw, closed, score, patched, normal = analyze(
            frame, _model, _ids, _mp, _landmarker, timestamp_ms
        )
    now = time.monotonic()
    logic.update(raw, closed, now)
    annotated = draw(frame, logic, raw, score, patched, normal)
    detections = [
        {"class_id": _ids["patched"], "class_name": "eye_patched", "confidence": confidence,
         "bbox": [x1, y1, x2, y2]}
        for (x1, y1, x2, y2), confidence in patched
    ] + [
        {"class_id": _ids["normal"], "class_name": "eye_normal", "confidence": confidence,
         "bbox": [x1, y1, x2, y2]}
        for (x1, y1, x2, y2), confidence in normal
    ]
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not encode annotated frame")
    return detections, base64.b64encode(encoded).decode("ascii"), logic


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(load_model)
    await asyncio.to_thread(load_landmarker)
    yield
    if _landmarker is not None:
        _landmarker.close()


app = FastAPI(title="PATCHY YOLO WebSocket", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "landmark_loaded": _landmarker is not None,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logic = SessionLogic()
    try:
        while True:
            message = await websocket.receive_json()
            frame_id = message.get("frame_id")
            encoded = message.get("image")
            if not isinstance(frame_id, int) or not isinstance(encoded, str):
                await websocket.send_json({"type": "error", "error": "Expected integer frame_id and base64 image"})
                continue
            try:
                frame_bytes = base64.b64decode(encoded, validate=True)
                frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Invalid image frame")
                detections, annotated, logic = await asyncio.to_thread(
                    process_frame, frame, logic, int(time.monotonic() * 1000)
                )
                message_text, level = logic.message()
                await websocket.send_json({
                    "type": "result",
                    "frame_id": frame_id,
                    "image": annotated,
                    "detections": detections,
                    "state": logic.state,
                    "message": message_text,
                    "level": level,
                    "therapy_seconds": round(logic.therapy_sec, 1),
                })
            except (ValueError, KeyError, binascii.Error) as error:
                await websocket.send_json({"type": "error", "frame_id": frame_id, "error": str(error)})
            except Exception as error:
                await websocket.send_json({"type": "error", "frame_id": frame_id, "error": f"Inference failed: {error}"})
    except WebSocketDisconnect:
        return
