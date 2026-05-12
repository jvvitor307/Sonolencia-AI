import os
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from src.config import (
    CNN_MODEL_PATH, PROCESSED_DIR, IMG_SIZE, BATCH_SIZE, RESULTS_DIR,
)
from src.landmarks import calculate_ear, calculate_mar, get_landmarks
from src.config import MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE


def evaluate_cnn_model():
    model = load_model(CNN_MODEL_PATH)
    val_dir = os.path.join(PROCESSED_DIR, "val")

    val_aug = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_aug.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    print("\n=== Avaliação do Modelo CNN ===")
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"Loss: {loss:.4f}")
    print(f"Acurácia: {accuracy:.4f}")

    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_labels)
    print(report_text)

    with open(os.path.join(RESULTS_DIR, 'final_classification_report.txt'), 'w') as f:
        f.write(f"Acurácia: {accuracy:.4f}\n")
        f.write(f"Loss: {loss:.4f}\n\n")
        f.write(report_text)

        f.write("\n\nMétricas por classe:\n")
        for label in class_labels:
            if label in report:
                f.write(f"  {label}:\n")
                f.write(f"    Precision: {report[label]['precision']:.4f}\n")
                f.write(f"    Recall: {report[label]['recall']:.4f}\n")
                f.write(f"    F1-Score: {report[label]['f1-score']:.4f}\n")

        f.write(f"\nMétricas agregadas:\n")
        f.write(f"  Macro F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")

    _plot_confusion_matrix(y_true, y_pred, class_labels)
    _plot_roc_curves(y_true, predictions, class_labels)
    _measure_inference_time(model)

    return report


def _plot_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title('Matriz de Confusão (Absoluta)', fontsize=14)
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predito')

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1])
    axes[1].set_title('Matriz de Confusão (Normalizada)', fontsize=14)
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Predito')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'final_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Matriz de confusão salva em {RESULTS_DIR}/final_confusion_matrix.png")


def _plot_roc_curves(y_true, predictions, class_labels):
    n_classes = len(class_labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, predictions[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
                label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax.set_title('Curvas ROC por Classe', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=150)
    plt.close()
    print(f"Curvas ROC salvas em {RESULTS_DIR}/roc_curves.png")


def _measure_inference_time(model, n_iterations=100):
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)

    for _ in range(10):
        model.predict(dummy_input, verbose=0)

    times = []
    for _ in range(n_iterations):
        start = time.time()
        model.predict(dummy_input, verbose=0)
        times.append(time.time() - start)

    times = np.array(times)
    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    fps = 1.0 / np.mean(times)

    filepath = os.path.join(RESULTS_DIR, 'inference_benchmark.txt')
    with open(filepath, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("BENCHMARK DE INFERENCIA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Iteracoes: {n_iterations}\n")
        f.write(f"Tempo medio: {mean_time:.2f} ± {std_time:.2f} ms\n")
        f.write(f"Tempo minimo: {np.min(times)*1000:.2f} ms\n")
        f.write(f"Tempo maximo: {np.max(times)*1000:.2f} ms\n")
        f.write(f"FPS equivalente: {fps:.1f}\n")
        f.write(f"\nStatus: {'ADEQUADO' if fps >= 15 else 'LENTO'} para tempo real\n")

    print(f"Benchmark: {mean_time:.2f} ms/imagem ({fps:.1f} FPS)")


def evaluate_ear_on_dataset():
    print("\n=== Avaliação do EAR no Dataset ===")
    val_dir = os.path.join(PROCESSED_DIR, "val")

    ear_values_closed = []
    ear_values_open = []

    for category in ["Closed", "Open"]:
        cat_dir = os.path.join(val_dir, category)
        if not os.path.exists(cat_dir):
            print(f"Diretório não encontrado: {cat_dir}")
            continue

        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:100]

        for f in files:
            img = cv2.imread(os.path.join(cat_dir, f))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = get_landmarks(rgb)
            if landmarks is not None:
                ear = calculate_ear(landmarks)
                if category == "Closed":
                    ear_values_closed.append(ear)
                else:
                    ear_values_open.append(ear)

    if ear_values_closed and ear_values_open:
        _plot_ear_distribution(ear_values_closed, ear_values_open)
    else:
        print("Dados insuficientes para plotar distribuição EAR.")


def _plot_ear_distribution(closed, open_eyes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(closed, bins=30, alpha=0.7, label='Olhos Fechados', color='red')
    axes[0].hist(open_eyes, bins=30, alpha=0.7, label='Olhos Abertos', color='green')
    axes[0].axvline(x=0.21, color='black', linestyle='--', label='Limiar EAR=0.21')
    axes[0].set_xlabel('EAR')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição EAR por Classe')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    data = [closed, open_eyes]
    bp = axes[1].boxplot(data, labels=['Fechados', 'Abertos'], patch_artist=True)
    bp['boxes'][0].set_facecolor('salmon')
    bp['boxes'][1].set_facecolor('lightgreen')
    axes[1].axhline(y=0.21, color='black', linestyle='--', label='Limiar')
    axes[1].set_ylabel('EAR')
    axes[1].set_title('Boxplot EAR por Classe')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ear_distribution.png'), dpi=150)
    plt.close()
    print(f"Distribuição EAR salva em {RESULTS_DIR}/ear_distribution.png")


if __name__ == "__main__":
    evaluate_cnn_model()
    evaluate_ear_on_dataset()
