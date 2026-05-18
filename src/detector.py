import os
import time
import collections
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import (
    RESULTS_DIR, EAR_THRESHOLD, MAR_THRESHOLD, EYES_CLOSED_TIME,
    CALIBRATION_DURATION, CALIBRATION_RATIO, CALIBRATION_MIN_SAMPLES,
)
from src.landmarks import (
    get_landmarks, calculate_ear, calculate_mar,
    detect_drowsiness, get_face_mesh_instance,
)


class DrowsinessDetector:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.face_mesh = get_face_mesh_instance()

        self.eyes_closed_start = None
        self.mar_counter = 0
        self.fps_history = collections.deque(maxlen=30)

        self.ear_history = collections.deque(maxlen=300)
        self.mar_history = collections.deque(maxlen=300)
        self.time_history = collections.deque(maxlen=300)

        self.alarm_active = False

        self.calibration_samples = []
        self.calibrated_ear_threshold = None
        self.calibration_done = False

    def _get_ear_threshold(self):
        if self.calibrated_ear_threshold is not None:
            return self.calibrated_ear_threshold
        return EAR_THRESHOLD

    def _update_calibration(self, ear, elapsed):
        if self.calibration_done:
            return

        if elapsed <= CALIBRATION_DURATION:
            self.calibration_samples.append(ear)
        elif elapsed > CALIBRATION_DURATION:
            if len(self.calibration_samples) >= CALIBRATION_MIN_SAMPLES:
                samples = np.array(self.calibration_samples)
                q25 = np.percentile(samples, 25)
                open_ear_samples = samples[samples >= q25]
                avg_ear = np.mean(open_ear_samples)
                self.calibrated_ear_threshold = round(avg_ear * CALIBRATION_RATIO, 3)
                print(f"Calibrado: EAR medio={avg_ear:.3f}, threshold={self.calibrated_ear_threshold}")
            else:
                print(f"Calibracao falhou ({len(self.calibration_samples)} amostras). Usando padrao: {EAR_THRESHOLD}")
            self.calibration_done = True

    def _draw_hud(self, frame, result, fps, elapsed):
        h, w = frame.shape[:2]
        ear_threshold = self._get_ear_threshold()
        overlay = frame.copy()

        if not self.calibration_done:
            progress = min(elapsed / CALIBRATION_DURATION, 1.0)
            bar_w = int(progress * w)
            cv2.rectangle(frame, (0, 0), (bar_w, 6), (0, 255, 255), -1)
            cv2.putText(frame, f"Calibrando... {progress*100:.0f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Mantenha os olhos abertos", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame

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

        ear_color = (0, 0, 255) if result['ear'] < ear_threshold else (0, 255, 0)
        cv2.putText(frame, f"EAR: {result['ear']:.3f} (>{ear_threshold:.3f})", (10, info_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)

        mar_color = (0, 0, 255) if result['mar'] > MAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"MAR: {result['mar']:.3f}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)

        if result['eye_closed'] and result['eyes_closed_duration'] > 0:
            dur_color = (0, 0, 255) if result['eyes_closed_duration'] >= EYES_CLOSED_TIME else (0, 165, 255)
            cv2.putText(frame, f"Olhos fechados: {result['eyes_closed_duration']:.1f}s",
                        (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dur_color, 2)

        ear_bar_w = int(min(result['ear'] / 0.4, 1.0) * 150)
        cv2.rectangle(frame, (w - 170, 20), (w - 170 + 150, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, 20), (w - 170 + ear_bar_w, 40), ear_color, -1)
        cv2.putText(frame, "EAR", (w - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        mar_bar_w = int(min(result['mar'] / 1.0, 1.0) * 150)
        cv2.rectangle(frame, (w - 170, 55), (w - 170 + 150, 75), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 170, 55), (w - 170 + mar_bar_w, 75), mar_color, -1)
        cv2.putText(frame, "MAR", (w - 170, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
        self.start_time = time.time()

        print("Pressione 'q' para sair, 'g' para gravar gráfico EAR.")
        print(f"Calibrando por {CALIBRATION_DURATION}s - mantenha os olhos abertos!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(rgb)

            result = {
                'ear': 0.3, 'mar': 0.0,
                'eyes_closed_duration': 0.0, 'mar_counter': 0,
                'eye_closed': False, 'mouth_open': False,
                'drowsy': False, 'reason': '',
                'eyes_closed_start': None,
            }

            elapsed = time.time() - self.start_time

            if landmarks is not None:
                ear = calculate_ear(landmarks)
                self._update_calibration(ear, elapsed)

                if self.calibration_done:
                    result = detect_drowsiness(
                        landmarks, self.eyes_closed_start, self.mar_counter,
                        ear_threshold=self._get_ear_threshold(),
                    )
                    self.eyes_closed_start = result['eyes_closed_start']
                    self.mar_counter = result['mar_counter']

                current_time = time.time() - self.start_time
                self.ear_history.append(ear)
                self.mar_history.append(calculate_mar(landmarks))
                self.time_history.append(current_time)
            else:
                self.eyes_closed_start = None
                cv2.putText(frame, "Rosto nao detectado", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)

            frame = self._draw_hud(frame, result, avg_fps, elapsed)

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
        ear_threshold = self._get_ear_threshold()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(times, ears, color='#2196F3', linewidth=1.5, label='EAR')
        axes[0].axhline(y=ear_threshold, color='red', linestyle='--', linewidth=1,
                        label=f'Limiar EAR ({ear_threshold})')
        axes[0].fill_between(times, 0, ear_threshold, alpha=0.15, color='red')
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
        ear_threshold = self._get_ear_threshold()
        filepath = os.path.join(RESULTS_DIR, 'inference_metrics.txt')
        with open(filepath, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("METRICAS DE INFERENCIA EM TEMPO REAL\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"FPS medio: {avg_fps:.2f}\n")
            f.write(f"Tempo de inferencia medio: {(1000 / max(avg_fps, 0.1)):.2f} ms\n")
            f.write(f"Limiar EAR: {ear_threshold} ({'calibrado' if self.calibrated_ear_threshold else 'padrao'})\n")
            f.write(f"Limiar MAR: {MAR_THRESHOLD}\n")
            f.write(f"Tempo para olhos fechados: {EYES_CLOSED_TIME}s\n")
            f.write(f"Frames consecutivos para boca: 10\n")
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
