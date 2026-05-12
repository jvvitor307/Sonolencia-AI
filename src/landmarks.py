import numpy as np
import mediapipe as mp
from src.config import (
    EAR_THRESHOLD, MAR_THRESHOLD,
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
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Para cada olho:
      p1 = canto externo, p4 = canto interno
      p2, p3 = pálpebra superior
      p5, p6 = pálpebra inferior
    """
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
    """
    MAR (Mouth Aspect Ratio) = ||p13-p14|| / ||p78-p308||

    p13, p14 = lábio superior e inferior (centro)
    p78 = canto esquerdo da boca
    p308 = canto direito da boca
    """
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_corner = landmarks[78]
    right_corner = landmarks[308]

    vertical = np.linalg.norm(upper_lip - lower_lip)
    horizontal = np.linalg.norm(left_corner - right_corner)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def detect_drowsiness(landmarks, ear_counter, mar_counter):
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)

    eye_closed = ear < EAR_THRESHOLD
    mouth_open = mar > MAR_THRESHOLD

    if eye_closed:
        ear_counter += 1
    else:
        ear_counter = max(0, ear_counter - 1)

    if mouth_open:
        mar_counter += 1
    else:
        mar_counter = max(0, mar_counter - 1)

    drowsy = False
    reason = ""

    if ear_counter >= 15:
        drowsy = True
        reason = "OLHOS FECHADOS"
    elif mar_counter >= 10:
        drowsy = True
        reason = "BOCEJO DETECTADO"

    return {
        'ear': ear,
        'mar': mar,
        'ear_counter': ear_counter,
        'mar_counter': mar_counter,
        'eye_closed': eye_closed,
        'mouth_open': mouth_open,
        'drowsy': drowsy,
        'reason': reason,
    }


def extract_eye_roi(frame, landmarks, eye_indices, padding=10):
    points = landmarks[eye_indices]
    x_min = int(np.min(points[:, 0])) - padding
    x_max = int(np.max(points[:, 0])) + padding
    y_min = int(np.min(points[:, 1])) - padding
    y_max = int(np.max(points[:, 1])) + padding

    h, w = frame.shape[:2]
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    y_min = max(0, y_min)
    y_max = min(h, y_max)

    return frame[y_min:y_max, x_min:x_max]


def get_face_mesh_instance():
    return FACE_MESH


def release_face_mesh():
    FACE_MESH.close()
