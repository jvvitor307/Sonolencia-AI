import os
import time
import collections
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.config import (
    CNN_MODEL_PATH, IMG_SIZE, RESULTS_DIR,
    EAR_THRESHOLD, MAR_THRESHOLD, MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE,
)
from src.landmarks import (
    get_landmarks, calculate_ear, calculate_mar,
    detect_drowsiness, extract_eye_roi, get_face_mesh_instance,
)


class DrowsinessDetector:
    def __init__(self, use_cnn=True, camera_id=0):
        self.use_cnn = use_cnn
        self.camera_id = camera_id
        self.model = None
        self.face_mesh = get_face_mesh_instance()

        self.ear_counter = 0
        self.mar_counter = 0
        self.fps_history = collections.deque(maxlen=30)

        self.ear_history = collections.deque(maxlen=300)
        self.mar_history = collections.deque(maxlen=300)
        self.time_history = collections.deque(maxlen=300)

        self.alarm_active = False
        self.start_time = time.time()

        if self.use_cnn:
            self._load_model()

    def _load_model(self):
        if os.path.exists(CNN_MODEL_PATH):
            self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
            print(f"Modelo CNN carregado: {CNN_MODEL_PATH}")
        else:
            print("Modelo CNN não encontrado. Usando apenas EAR/MAR geométrico.")
            self.use_cnn = False

    def _predict_eye_state(self, eye_roi):
        if self.model is None:
            return None, 0.0

        try:
            eye_resized = cv2.resize(eye_roi, IMG_SIZE)
            eye_normalized = eye_resized.astype(np.float32) / 255.0
            eye_batch = np.expand_dims(eye_normalized, axis=0)
            prediction = self.model.predict(eye_batch, verbose=0)[0]

            class_names = ['Closed', 'Open', 'no_yawn', 'yawn']
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

            is_closed = class_idx == 0
            return is_closed, float(confidence)
        except Exception:
            return None, 0.0

    def _draw_hud(self, frame, result, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        if result['drowsy']:
            color = (0, 0, 255)
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            text = f"FADIGA: {result['reason']}"
            font_scale = min(w / 500, 1.5)
            thickness = max(int(font_scale * 2), 2)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.putText(frame, text,
                        ((w - text_size[0]) // 2, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            self.alarm_active = True
        else:
            self.alarm_active = False

        info_y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ear_color = (0, 0, 255) if result['ear'] < EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"EAR: {result['ear']:.3f}", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)

        mar_color = (0, 0, 255) if result['mar'] > MAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"MAR: {result['mar']:.3f}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)

        ear_bar_w = int(min(result['ear'] / 0.4, 1.0) * 150)
        cv2.rectangle(frame, (w - 170, 20), (w - 170 + 150, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, 20), (w - 170 + ear_bar_w, 40), ear_color, -1)
        cv2.putText(frame, "EAR", (w - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        mar_bar_w = int(min(result['mar'] / 1.0, 1.0) * 150)
        cv2.rectangle(frame, (w - 170, 55), (w - 170 + 150, 75), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, 55), (w - 170 + mar_bar_w, 75), mar_color, -1)
        cv2.putText(frame, "MAR", (w - 170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if self.use_cnn and result.get('cnn_closed') is not None:
            cnn_text = "CNN: FECHADO" if result['cnn_closed'] else "CNN: ABERTO"
            cnn_color = (0, 0, 255) if result['cnn_closed'] else (0, 255, 0)
            cv2.putText(frame, f"{cnn_text} ({result['cnn_conf']:.2f})",
                        (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cnn_color, 2)

        return frame

    def run(self, display=True, save_video=False, output_path=None):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            return

        writer = None
        if save_video:
            if output_path is None:
                output_path = os.path.join(RESULTS_DIR, "drowsiness_detection.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        prev_time = time.time()
        frame_count = 0

        print("Pressione 'q' para sair, 'g' para gravar gráfico EAR.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(rgb)

            result = {
                'ear': 0.3, 'mar': 0.0,
                'ear_counter': 0, 'mar_counter': 0,
                'eye_closed': False, 'mouth_open': False,
                'drowsy': False, 'reason': '',
                'cnn_closed': None, 'cnn_conf': 0.0,
            }

            if landmarks is not None:
                result = detect_drowsiness(landmarks, self.ear_counter, self.mar_counter)
                self.ear_counter = result['ear_counter']
                self.mar_counter = result['mar_counter']

                current_time = time.time() - self.start_time
                self.ear_history.append(result['ear'])
                self.mar_history.append(result['mar'])
                self.time_history.append(current_time)

                if self.use_cnn:
                    try:
                        left_eye = extract_eye_roi(frame, landmarks, MEDIAPIPE_LEFT_EYE)
                        right_eye = extract_eye_roi(frame, landmarks, MEDIAPIPE_RIGHT_EYE)
                        eyes = [e for e in [left_eye, right_eye] if e.size > 0]
                        if eyes:
                            eye = max(eyes, key=lambda e: e.shape[0] * e.shape[1])
                            cnn_closed, cnn_conf = self._predict_eye_state(eye)
                            result['cnn_closed'] = cnn_closed
                            result['cnn_conf'] = cnn_conf

                            if cnn_closed is not None and cnn_closed and cnn_conf > 0.7:
                                if not result['drowsy']:
                                    self.ear_counter += 3
                                    result['ear_counter'] = self.ear_counter
                                    if self.ear_counter >= 15:
                                        result['drowsy'] = True
                                        result['reason'] = "OLHOS FECHADOS (CNN)"
                    except Exception:
                        pass
            else:
                cv2.putText(frame, "Rosto nao detectado", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)

            frame = self._draw_hud(frame, result, avg_fps)

            if display:
                cv2.imshow('Drowsiness Detection', frame)

            if writer:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self._save_ear_plot()

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self._save_ear_plot()
        self._save_metrics_report(avg_fps)

    def _save_ear_plot(self):
        if len(self.time_history) < 2:
            return

        times = list(self.time_history)
        ears = list(self.ear_history)
        mars = list(self.mar_history)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(times, ears, color='#2196F3', linewidth=1.5, label='EAR')
        axes[0].axhline(y=EAR_THRESHOLD, color='red', linestyle='--', linewidth=1,
                        label=f'Limiar EAR ({EAR_THRESHOLD})')
        axes[0].fill_between(times, 0, EAR_THRESHOLD, alpha=0.15, color='red')
        axes[0].set_ylabel('EAR (Eye Aspect Ratio)', fontsize=12)
        axes[0].set_title('Razão de Aspecto do Olho em Tempo Real', fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 0.45)

        axes[1].plot(times, mars, color='#FF9800', linewidth=1.5, label='MAR')
        axes[1].axhline(y=MAR_THRESHOLD, color='red', linestyle='--', linewidth=1,
                        label=f'Limiar MAR ({MAR_THRESHOLD})')
        axes[1].fill_between(times, MAR_THRESHOLD, max(mars) if mars else 1, alpha=0.15, color='red')
        axes[1].set_ylabel('MAR (Mouth Aspect Ratio)', fontsize=12)
        axes[1].set_xlabel('Tempo (s)', fontsize=12)
        axes[1].set_title('Razão de Aspecto da Boca em Tempo Real', fontsize=14)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(RESULTS_DIR, 'ear_mar_realtime.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Gráfico EAR/MAR salvo: {filepath}")

    def _save_metrics_report(self, avg_fps):
        filepath = os.path.join(RESULTS_DIR, 'inference_metrics.txt')
        with open(filepath, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("METRICAS DE INFERENCIA EM TEMPO REAL\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"FPS medio: {avg_fps:.2f}\n")
            f.write(f"Tempo de inferencia medio: {(1000 / max(avg_fps, 0.1)):.2f} ms\n")
            f.write(f"Limiar EAR: {EAR_THRESHOLD}\n")
            f.write(f"Limiar MAR: {MAR_THRESHOLD}\n")
            f.write(f"Frames consecutivos para olhos: 15\n")
            f.write(f"Frames consecutivos para boca: 10\n")
            f.write(f"CNN utilizada: {'Sim' if self.use_cnn else 'Nao'}\n")
            f.write(f"\nTotal de frames processados: {len(self.ear_history)}\n")
            if self.ear_history:
                ears = list(self.ear_history)
                f.write(f"\nEAR - Min: {min(ears):.3f}, Max: {max(ears):.3f}, "
                        f"Medio: {np.mean(ears):.3f}\n")
            if self.mar_history:
                mars = list(self.mar_history)
                f.write(f"MAR - Min: {min(mars):.3f}, Max: {max(mars):.3f}, "
                        f"Medio: {np.mean(mars):.3f}\n")
        print(f"Métricas de inferência salvas: {filepath}")
