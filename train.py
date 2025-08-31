import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths and constants
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "training_data")
VAL_SPLIT = 0.2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# Ensure GPU memory growth if GPU is available
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Dataset loaders

def make_datasets():
    # Use image_dataset_from_directory to mirror app preprocessing (grayscale 300x300)
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )

    # Normalization to [0,1] to match app.py
    def to_float(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(to_float, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(to_float, num_parallel_calls=AUTOTUNE)

    # Data augmentation (texture-friendly)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.06),  # ~±20 deg
            layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
            layers.RandomContrast(0.1),
            # Add small noise
            layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)),
            layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
        ],
        name="augmentation",
    )

    def apply_augmentation(x, y):
        return data_augmentation(x, training=True), y

    train_ds = train_ds.map(apply_augmentation, num_parallel_calls=AUTOTUNE)

    # Caching and prefetch
    train_ds = train_ds.shuffle(8 * BATCH_SIZE, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights():
    # Count files per class inside TRAIN_DIR
    class_counts = {}
    if not os.path.isdir(TRAIN_DIR):
        return None

    for class_name in sorted(os.listdir(TRAIN_DIR)):
        class_path = os.path.join(TRAIN_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        count = 0
        for root, _, files in os.walk(class_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    count += 1
        if count > 0:
            class_counts[class_name] = count

    if len(class_counts) < 2:
        return None

    # Map class names to 0/1, try to follow notebook mapping if folders are named
    # normal -> 0, cracked -> 1; otherwise fall back to alphabetical order
    mapping = {}
    names = sorted(list(class_counts.keys()))
    if "normal" in class_counts and "cracked" in class_counts:
        mapping = {"normal": 0, "cracked": 1}
    else:
        mapping = {names[0]: 0, names[1]: 1}

    counts_by_index = {mapping[k]: v for k, v in class_counts.items() if k in mapping}

    total = sum(counts_by_index.values())
    if total == 0 or len(counts_by_index) != 2:
        return None

    # Inverse frequency weighting
    weights = {}
    for idx, cnt in counts_by_index.items():
        weights[idx] = total / (2.0 * cnt)
    return weights


# Model

def build_model(input_shape=(300, 300, 1), l2=1e-4, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="tire_grayscale_cnn")

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_ds, val_ds = make_datasets()

    model = build_model()
    model.summary()

    # Callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(PROJECT_DIR, "model3.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1
    )

    class_weights = compute_class_weights()
    if class_weights:
        print("Using class weights:", class_weights)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model as well (best already saved by checkpoint)
    final_path = os.path.join(PROJECT_DIR, "model3_final.keras")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
