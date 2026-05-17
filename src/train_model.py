import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from src.config import (
    PROCESSED_DIR, MODEL_DIR, CNN_MODEL_PATH,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, RESULTS_DIR
)


def create_data_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
    )

    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(PROCESSED_DIR, "train")
    val_dir = os.path.join(PROCESSED_DIR, "val")
    test_dir = os.path.join(PROCESSED_DIR, "test")

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )

    val_gen = val_aug.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    test_gen = None
    if os.path.isdir(test_dir):
        subdirs = [d for d in os.listdir(test_dir)
                   if os.path.isdir(os.path.join(test_dir, d))]
        if any(len(os.listdir(os.path.join(test_dir, d))) > 0 for d in subdirs):
            test_gen = val_aug.flow_from_directory(
                test_dir,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=False,
            )

    print(f"Classes: {train_gen.class_indices}")
    print(f"Amostras de treino: {train_gen.samples}")
    print(f"Amostras de validação: {val_gen.samples}")
    if test_gen:
        print(f"Amostras de teste: {test_gen.samples}")

    return train_gen, val_gen, test_gen


def build_mobilenetv2(num_classes=4):
    base = applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )

    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model, base


def build_custom_cnn(num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def fine_tune_model(model, base_model, unfreeze_from=100):
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(model_type="mobilenetv2"):
    train_gen, val_gen, test_gen = create_data_generators()
    num_classes = len(train_gen.class_indices)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            CNN_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
        ),
    ]

    if model_type == "mobilenetv2":
        model, base = build_mobilenetv2(num_classes)
    else:
        model = build_custom_cnn(num_classes)

    model.summary()

    print("\n=== Fase 1: Treino com backbone congelado ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
    )

    if model_type == "mobilenetv2":
        print("\n=== Fase 2: Fine-tuning ===")
        model = fine_tune_model(model, base)
        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS // 2,
            callbacks=callbacks,
        )
        history.history['accuracy'].extend(history_ft.history['accuracy'])
        history.history['val_accuracy'].extend(history_ft.history['val_accuracy'])
        history.history['loss'].extend(history_ft.history['loss'])
        history.history['val_loss'].extend(history_ft.history['val_loss'])

    plot_training_history(history)

    eval_gen = test_gen if test_gen else val_gen
    evaluate_model(model, eval_gen, "test" if test_gen else "val")

    model.save(CNN_MODEL_PATH)
    print(f"\nModelo salvo em: {CNN_MODEL_PATH}")

    return model, history


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Treino', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    axes[0].set_title('Acurácia por Época', fontsize=14)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Acurácia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Treino', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validação', linewidth=2)
    axes[1].set_title('Loss por Época', fontsize=14)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Gráfico de treino salvo em {RESULTS_DIR}/training_history.png")


def evaluate_model(model, gen, split_name="test"):
    gen.reset()
    predictions = model.predict(gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = gen.classes

    class_labels = list(gen.class_indices.keys())

    print(f"\n=== Relatório de Classificação ({split_name}) ===")
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(report)

    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Matriz de Confusão ({split_name})', fontsize=14)
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    print(f"Matriz de confusão salva em {RESULTS_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "mobilenetv2"
    train_model(model_type)
