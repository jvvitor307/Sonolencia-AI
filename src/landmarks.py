import time
import numpy as np
import mediapipe as mp
from src.config import (
    EAR_THRESHOLD, MAR_THRESHOLD, EYES_CLOSED_TIME, CONSEC_FRAMES_MAR,
    MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE,
)


mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_landmarks(frame_rgb):
    results = FACE_MESH.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = frame_rgb.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
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


def detect_drowsiness(landmarks, eyes_closed_start, mar_counter):
    now = time.time()
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)

    eye_closed = ear < EAR_THRESHOLD
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
    return FACE_MESH


def release_face_mesh():
    FACE_MESH.close()
