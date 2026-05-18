import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from src.config import (
    BASE_DIR, EAR_THRESHOLD, MAR_THRESHOLD, EYES_CLOSED_TIME, CONSEC_FRAMES_MAR,
    MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE,
)

MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "face_landmarker.task")

_face_landmarker = None


def _get_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def get_landmarks(frame_rgb):
    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = frame_rgb.shape[:2]
    landmarks = result.face_landmarks[0]
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks])


def calculate_ear(landmarks):
    left_ear = _single_eye_ear(landmarks, MEDIAPIPE_LEFT_EYE)
    right_ear = _single_eye_ear(landmarks, MEDIAPIPE_RIGHT_EYE)
    return (left_ear + right_ear) / 2.0


def _single_eye_ear(landmarks, indices):
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.3

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_corner = landmarks[78]
    right_corner = landmarks[308]

    vertical = np.linalg.norm(upper_lip - lower_lip)
    horizontal = np.linalg.norm(left_corner - right_corner)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def detect_drowsiness(landmarks, eyes_closed_start, mar_counter, ear_threshold=None):
    now = time.time()
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)

    threshold = ear_threshold if ear_threshold is not None else EAR_THRESHOLD
    eye_closed = ear < threshold
    mouth_open = mar > MAR_THRESHOLD

    if eye_closed:
        if eyes_closed_start is None:
            eyes_closed_start = now
    else:
        eyes_closed_start = None

    if mouth_open:
        mar_counter += 1
    else:
        mar_counter = max(0, mar_counter - 1)

    drowsy = False
    reason = ""

    eyes_closed_duration = 0.0
    if eyes_closed_start is not None:
        eyes_closed_duration = now - eyes_closed_start

    if eyes_closed_duration >= EYES_CLOSED_TIME:
        drowsy = True
        reason = "DORMINDO"
    elif mar_counter >= CONSEC_FRAMES_MAR:
        drowsy = True
        reason = "BOCEJO DETECTADO"

    return {
        'ear': ear,
        'mar': mar,
        'eyes_closed_duration': eyes_closed_duration,
        'mar_counter': mar_counter,
        'eye_closed': eye_closed,
        'mouth_open': mouth_open,
        'drowsy': drowsy,
        'reason': reason,
        'eyes_closed_start': eyes_closed_start,
    }


def get_face_mesh_instance():
    return _get_landmarker()


def release_face_mesh():
    global _face_landmarker
    if _face_landmarker is not None:
        _face_landmarker.close()
        _face_landmarker = None
