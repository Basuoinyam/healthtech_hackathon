
"""
Lazy-eye (amblyopia) patch-session monitor.

YOLO classes:
    eye_normal
    eye_patched

Timer rules:
    1 patched + 1 normal -> therapy timer runs.
    1 normal only -> assume the other eye is patched.
    Both eyes visible -> timer paused.
    Both eyes patched -> timer paused.
    No eye detection -> timer paused.
    Uncovered eye closed for 3 seconds -> alert and pause.

Event timestamps:
    t = elapsed seconds from the beginning of the session.
"""

import json
import os
import time
from collections import Counter, deque

import cv2
import numpy as np


# ----------------------------- settings -----------------------------

CONFIRM_FRAMES = 50
CONFIRM_RATIO = 0.8

CLOSED_SECONDS = 2.0
BLINK_THRESHOLD = 0.5
BLINK_SMOOTH_FRAMES = 5

YOLO_CONF = 0.2
YOLO_IMGSZ = 640

ALERT_REPEAT_SEC = 10
MAX_DT = 0.5


# -------------------------- runtime config ---------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.getenv(
    "YOLO_WEIGHTS",
    os.path.join(PROJECT_DIR, "models", "best (1).pt")
)
VIDEO_SOURCE = 0
LANDMARKER_PATH = os.getenv(
    "LANDMARKER_PATH",
    os.path.join(PROJECT_DIR, "face_landmarker.task")
)

OUTPUT_VIDEO = None  # "annotated_output.mp4"
SHOW_WINDOW = True
SESSION_LOG = "session_log.json"

YOLO_DEVICE = "cpu"  # Use 0 for NVIDIA GPU


# ----------------------------- states -----------------------------

THERAPY = "THERAPY"
BOTH_OPEN = "BOTH_OPEN"
BOTH_PATCHED = "BOTH_PATCHED"
LOST = "LOST"


# MediaPipe face mesh eye landmarks.
EYES = {
    "right": {
        "ids": [
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 173, 157, 158, 159, 160, 161, 246
        ],
        "blink": "eyeBlinkRight",
        "ear": ((159, 145), (158, 153), (160, 144), (33, 133))
    },

    "left": {
        "ids": [
            362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398
        ],
        "blink": "eyeBlinkLeft",
        "ear": ((386, 374), (385, 380), (387, 373), (362, 263))
    }
}


# =====================================================================
# State logic
# =====================================================================

def raw_state_from(n_patched, n_normal, eye_closed):

    if n_patched >= 2:
        return BOTH_PATCHED

    if n_patched == 1:

        if n_normal >= 1 or eye_closed:
            return THERAPY

        return LOST

    if n_normal >= 2:
        return BOTH_OPEN

    if n_normal == 1:
        # Assume the other eye is covered by a patch
        # that YOLO was not trained to detect.
        return THERAPY

    return LOST


# =====================================================================
# Session logic and timer
# =====================================================================

class SessionLogic:

    def __init__(
        self,
        confirm_frames=CONFIRM_FRAMES,
        ratio=CONFIRM_RATIO,
        closed_seconds=CLOSED_SECONDS,
        smooth=BLINK_SMOOTH_FRAMES
    ):

        self.window = deque(maxlen=confirm_frames)
        self.ratio = ratio

        self.closed_seconds = closed_seconds
        self.closed_hist = deque(maxlen=smooth)

        self.state = LOST
        self.started = False

        self.therapy_sec = 0.0
        self.paused_sec = 0.0

        self.closed_since = None
        self.closed_alert = False

        self.last_t = None

        # Session start time from monotonic clock.
        self.session_start_t = None

        self.events = []

    def _event(self, name, t):

        # Elapsed seconds from session start.
        elapsed_sec = max(
            0.0,
            t - self.session_start_t
        )

        self.events.append({

            "t": round(elapsed_sec, 2),

            "event": name,

            "therapy_sec": round(self.therapy_sec, 1)

        })

    def update(self, raw_state, eye_closed, t):

        # Initialize session clock on first processed frame.
        if self.session_start_t is None:
            self.session_start_t = t

        # Calculate elapsed time using monotonic clock.
        dt = (
            0.0
            if self.last_t is None
            else min(max(t - self.last_t, 0.0), MAX_DT)
        )

        self.last_t = t

        # -------------------------------------------------------------
        # 1. Temporal voting
        # -------------------------------------------------------------

        self.window.append(raw_state)

        if len(self.window) == self.window.maxlen:

            st, cnt = Counter(self.window).most_common(1)[0]

            if (
                cnt / len(self.window) >= self.ratio
                and st != self.state
            ):

                self.state = st

                self._event(
                    f"state->{st}",
                    t
                )

        # -------------------------------------------------------------
        # 2. Eye closure detection
        # -------------------------------------------------------------

        self.closed_hist.append(
            bool(
                eye_closed
                and raw_state == THERAPY
            )
        )

        closed = np.mean(self.closed_hist) >= 0.6

        if self.state == THERAPY and closed:

            if self.closed_since is None:
                self.closed_since = t

            if (
                not self.closed_alert
                and t - self.closed_since >= self.closed_seconds
            ):

                self.closed_alert = True

                self._event(
                    "eye_closed_alert",
                    t
                )

        else:

            if self.closed_alert:

                self._event(
                    "eye_reopened",
                    t
                )

            self.closed_since = None
            self.closed_alert = False

        # -------------------------------------------------------------
        # 3. Therapy timer
        # -------------------------------------------------------------

        if self.running:

            self.therapy_sec += dt
            self.started = True

        elif self.started:

            self.paused_sec += dt

    @property
    def running(self):

        return (
            self.state == THERAPY
            and not self.closed_alert
        )

    def message(self):

        if self.state == THERAPY:

            if self.closed_alert:
                return "Open your other eye!", "alert"

            return "Therapy running", "ok"

        if len(self.window) < self.window.maxlen and not self.started:
            return "Detecting...", "info"

        level = "alert" if self.started else "warn"

        if self.state == BOTH_OPEN:

            return (
                "Both eyes visible - put the patch on one eye",
                level
            )

        if self.state == BOTH_PATCHED:

            return (
                "Both eyes covered - uncover one eye",
                level
            )

        return (
            "Face not visible or patch not detected - "
            "face the camera / put the patch on",
            level
        )

    def summary(self):

        return {

            "therapy_seconds": round(
                self.therapy_sec,
                1
            ),

            "paused_seconds": round(
                self.paused_sec,
                1
            ),

            "events": self.events

        }


# =====================================================================
# Helper functions
# =====================================================================

def nearest_side(point, centers):

    return min(
        centers,
        key=lambda k: float(
            np.linalg.norm(
                np.asarray(point) - centers[k]
            )
        )
    )


def fmt_time(sec):

    sec = int(sec)

    hours = sec // 3600
    minutes = (sec % 3600) // 60
    seconds = sec % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# =====================================================================
# MediaPipe
# =====================================================================

def import_mediapipe():

    try:

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

    except NameError:

        import sys
        import types

        for k in [
            k for k in sys.modules
            if k == "mediapipe" or k.startswith("mediapipe.")
        ]:

            del sys.modules[k]

        sys.modules[
            "mediapipe.tasks.python.audio"
        ] = types.ModuleType(
            "mediapipe.tasks.python.audio"
        )

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

    return mp, mp_python, vision


def create_landmarker(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"MediaPipe model file not found: {path}\n"
            "Update LANDMARKER_PATH."
        )

    mp, mp_python, vision = import_mediapipe()

    options = vision.FaceLandmarkerOptions(

        base_options=mp_python.BaseOptions(
            model_asset_path=path
        ),

        running_mode=vision.RunningMode.VIDEO,

        num_faces=1,

        output_face_blendshapes=True
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    return mp, landmarker


def eye_info(mp, landmarker, frame_bgr, ts_ms):

    rgb = np.ascontiguousarray(
        cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )
    )

    result = landmarker.detect_for_video(

        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        ),

        ts_ms
    )

    if not result.face_landmarks:
        return None

    h, w = frame_bgr.shape[:2]

    points = np.array(
        [
            [landmark.x * w, landmark.y * h]
            for landmark in result.face_landmarks[0]
        ],
        dtype=np.float32
    )

    centers = {

        side: points[data["ids"]].mean(0)

        for side, data in EYES.items()

    }

    if result.face_blendshapes:

        blendshapes = {

            category.category_name: category.score

            for category in result.face_blendshapes[0]

        }

        blink = {

            side: float(
                blendshapes.get(data["blink"], 0.0)
            )

            for side, data in EYES.items()

        }

    else:

        blink = {}

        for side, data in EYES.items():

            (a1, a2), (b1, b2), (c1, c2), (h1, h2) = data["ear"]

            vertical = sum(

                np.linalg.norm(
                    points[i] - points[j]
                )

                for i, j in (
                    (a1, a2),
                    (b1, b2),
                    (c1, c2)
                )

            ) / 3

            horizontal = (
                np.linalg.norm(
                    points[h1] - points[h2]
                ) + 1e-6
            )

            ear = vertical / horizontal

            blink[side] = float(
                np.clip(
                    1 - ear / 0.28,
                    0,
                    1
                )
            )

    return centers, blink


# =====================================================================
# YOLO + MediaPipe analysis
# =====================================================================

def analyze(
    frame,
    yolo,
    ids,
    mp,
    landmarker,
    ts_ms
):

    result = yolo.predict(

        frame,

        imgsz=YOLO_IMGSZ,

        conf=YOLO_CONF,

        agnostic_nms=True,

        max_det=2,

        device=YOLO_DEVICE,

        verbose=False

    )[0]

    patched = []
    normal = []

    for box in result.boxes:

        x1, y1, x2, y2 = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        item = (
            (x1, y1, x2, y2),
            float(box.conf)
        )

        if int(box.cls) == ids["patched"]:

            patched.append(item)

        elif int(box.cls) == ids["normal"]:

            normal.append(item)

    eye_closed = False
    score = None

    # Case 1:
    # One patched eye detected.
    # Monitor the opposite uncovered eye.

    if len(patched) == 1:

        if mp is not None and landmarker is not None:
            info = eye_info(
                mp,
                landmarker,
                frame,
                ts_ms
            )
        else:
            info = None
        if info:

            centers, blink = info

            (x1, y1, x2, y2), _ = patched[0]

            covered_side = nearest_side(

                (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                ),

                centers
            )

            uncovered_side = (
                "left"
                if covered_side == "right"
                else "right"
            )

            score = blink[uncovered_side]

            eye_closed = score >= BLINK_THRESHOLD

    # Case 2:
    # One normal eye and no patched eye detected.
    # Assume the other eye is patched.
    # Monitor the detected visible eye.

    elif len(normal) == 1 and len(patched) == 0:

        if mp is not None and landmarker is not None:
            info = eye_info(
                mp,
                landmarker,
                frame,
                ts_ms
            )
        else:
            info = None
        if info:

            centers, blink = info

            (x1, y1, x2, y2), _ = normal[0]

            visible_side = nearest_side(

                (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2
                ),

                centers
            )

            score = blink[visible_side]

            eye_closed = score >= BLINK_THRESHOLD

    raw_state = raw_state_from(

        len(patched),

        len(normal),

        eye_closed

    )

    return (
        raw_state,
        eye_closed,
        score,
        patched,
        normal
    )


# =====================================================================
# Drawing and alerts
# =====================================================================

LEVEL_COLOR = {

    "ok": (0, 190, 0),

    "info": (200, 200, 200),

    "warn": (0, 165, 255),

    "alert": (0, 0, 255)

}


def draw(
    frame,
    logic,
    raw,
    score,
    patched,
    normal
):

    for (x1, y1, x2, y2), confidence in patched:

        cv2.rectangle(

            frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            (0, 0, 255),

            2

        )

        cv2.putText(

            frame,

            f"patched {confidence:.2f}",

            (int(x1), max(14, int(y1) - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 0, 255),

            1

        )

    for (x1, y1, x2, y2), confidence in normal:

        cv2.rectangle(

            frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            (0, 200, 0),

            2

        )

        cv2.putText(

            frame,

            f"normal {confidence:.2f}",

            (int(x1), max(14, int(y1) - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 200, 0),

            1

        )

    message, level = logic.message()

    color = LEVEL_COLOR[level]

    h, w = frame.shape[:2]

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 64),
        (30, 30, 30),
        -1
    )

    cv2.putText(

        frame,

        f"Timer {fmt_time(logic.therapy_sec)}",

        (10, 26),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )

    cv2.putText(

        frame,

        message[:90],

        (10, 54),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        color,

        2

    )

    debug_text = (
        f"raw={raw} state={logic.state}"
        + (
            f" blink={score:.2f}"
            if score is not None
            else ""
        )
    )

    cv2.putText(

        frame,

        debug_text,

        (10, h - 10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        1

    )

    return frame


def beep():

    try:

        import winsound

        winsound.Beep(1000, 300)

    except Exception:

        print("\a", end="", flush=True)


# =====================================================================
# Main loop
# =====================================================================

def main():

    from ultralytics import YOLO

    if not os.path.isfile(YOLO_WEIGHTS):

        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_WEIGHTS}"
        )

    yolo = YOLO(YOLO_WEIGHTS)

    model_names = yolo.names

    if isinstance(model_names, list):

        model_names = {
            i: name
            for i, name in enumerate(model_names)
        }

    name_to_id = {

        str(name): int(class_id)

        for class_id, name in model_names.items()

    }

    required_classes = (
        "eye_patched",
        "eye_normal"
    )

    missing = [

        name
        for name in required_classes
        if name not in name_to_id

    ]

    if missing:

        raise ValueError(
            f"YOLO model is missing classes: {missing}. "
            f"Model classes: {list(name_to_id.keys())}"
        )

    ids = {

        "patched": name_to_id["eye_patched"],

        "normal": name_to_id["eye_normal"]

    }

    mp, landmarker = create_landmarker(
        LANDMARKER_PATH
    )

    is_cam = (

        isinstance(VIDEO_SOURCE, int)

        or (
            isinstance(VIDEO_SOURCE, str)
            and VIDEO_SOURCE.isdigit()
        )

    )

    source = (
        int(VIDEO_SOURCE)
        if is_cam
        else VIDEO_SOURCE
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        landmarker.close()

        raise RuntimeError(
            f"Could not open video source: {VIDEO_SOURCE}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    logic = SessionLogic()

    writer = None

    idx = 0
    last_ts = -1

    last_alert_msg = None
    last_beep = 0.0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # Webcam uses monotonic time.
            # Video files use their own frame timeline.

            t = (
                time.monotonic()
                if is_cam
                else idx / fps
            )

            ts_ms = max(
                last_ts + 1,
                int(t * 1000)
            )

            last_ts = ts_ms

            raw, closed, score, patched, normal = analyze(

                frame,

                yolo,

                ids,

                mp,

                landmarker,

                ts_ms

            )

            logic.update(
                raw,
                closed,
                t
            )

            message, level = logic.message()

            if level == "alert" and (

                message != last_alert_msg

                or (
                    time.monotonic() - last_beep
                    >= ALERT_REPEAT_SEC
                )

            ):

                print(
                    f"[ALERT {fmt_time(logic.therapy_sec)}] "
                    f"{message}"
                )

                beep()

                last_beep = time.monotonic()

            last_alert_msg = (
                message
                if level == "alert"
                else None
            )

            output_frame = draw(

                frame,

                logic,

                raw,

                score,

                patched,

                normal

            )

            if OUTPUT_VIDEO:

                if writer is None:

                    h, w = output_frame.shape[:2]

                    writer = cv2.VideoWriter(

                        OUTPUT_VIDEO,

                        cv2.VideoWriter_fourcc(*"mp4v"),

                        fps,

                        (w, h)

                    )

                writer.write(output_frame)

            if SHOW_WINDOW:

                cv2.imshow(
                    "Patch session",
                    output_frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            idx += 1

    finally:

        cap.release()

        if writer is not None:
            writer.release()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        landmarker.close()

    # Save session summary.
    summary = logic.summary()

    with open(
        SESSION_LOG,
        "w",
        encoding="utf-8"
    ) as log_file:

        json.dump(
            summary,
            log_file,
            indent=2,
            ensure_ascii=False
        )

    print(

        f"Session finished: "

        f"therapy {fmt_time(summary['therapy_seconds'])}, "

        f"paused {fmt_time(summary['paused_seconds'])}, "

        f"log -> {SESSION_LOG}"

    )


if __name__ == "__main__":

    main()
