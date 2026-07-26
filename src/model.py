"""
model.py
Model architecture, training, evaluation and retraining for the Bean pipeline.
"""

import os
import json
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from .preprocessing import (
    IMG_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    TRAIN_DIR,
    TEST_DIR,
    build_augmenter,
    load_dataset_from_dir,
)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "beans_model.keras")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "20"))
ACC_FLOOR = float(os.getenv("ACC_FLOOR", "0.80"))


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def build_model() -> Tuple[tf.keras.Model, tf.keras.Model]:
    """MobileNetV2 transfer-learning model. Returns (model, base) so the caller
    can unfreeze `base` for fine-tuning."""
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # stage 1: frozen backbone

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmenter()(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)                                   # regularization
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)  # L2
    )(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="beans_mobilenetv2"), base


UNFREEZE_LAST = 60   # how many backbone layers to fine-tune
FINETUNE_LR = 1e-4   # 10x below the head LR, 10x above a timid 1e-5


def default_callbacks(patience: int = 8):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7, verbose=1
        ),
    ]


# --------------------------------------------------------------------------- #
# Training from scratch (two-stage: head, then fine-tune)
# --------------------------------------------------------------------------- #
def train(train_ds, val_ds, epochs_head: int = 25, epochs_ft: int = 30) -> tf.keras.Model:
    model, base = build_model()

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, callbacks=default_callbacks())

    # Stage 2 — fine-tune the top of the backbone.
    # NOTE: build_model() calls base(x, training=False), which pins the backbone's
    # BatchNorm layers to inference mode. Setting base.trainable = True does not
    # override that — this is the recommended Keras fine-tuning pattern and stops
    # BN from recomputing statistics on our small dataset.
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(FINETUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft, callbacks=default_callbacks())

    save_model(model)
    return model


def save_model(model: tf.keras.Model, path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f)
    return path


def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model at {path}. Train one first (run the notebook) and commit it to models/."
        )
    return tf.keras.models.load_model(path)


# --------------------------------------------------------------------------- #
# Evaluation — the 4+ metrics required by the rubric
# --------------------------------------------------------------------------- #
def evaluate(model: tf.keras.Model, test_ds) -> dict:
    y_true, y_prob = [], []
    for X, y in test_ds:
        y_prob.append(model.predict(X, verbose=0))
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "evaluated_at": time.time(),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load_metrics() -> dict:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# Retraining — this is what the UI button / API endpoint calls
# --------------------------------------------------------------------------- #
def should_retrain(num_new_images: int, current_acc: float = None) -> Tuple[bool, str]:
    """Trigger rule: enough fresh data, or live accuracy has decayed."""
    if num_new_images >= RETRAIN_THRESHOLD:
        return True, f"{num_new_images} new images >= threshold of {RETRAIN_THRESHOLD}"
    if current_acc is not None and current_acc < ACC_FLOOR:
        return True, f"accuracy {current_acc:.3f} below floor of {ACC_FLOOR}"
    return False, f"no trigger ({num_new_images} new images, threshold {RETRAIN_THRESHOLD})"


def retrain(epochs: int = 8, lr: float = 1e-5) -> dict:
    """
    Fine-tune the EXISTING custom model (used as the pre-trained starting point)
    on everything currently in data/train/ — original data plus uploaded images.
    """
    started = time.time()

    model = load_model()  # our own previously-trained model = the pretrained base

    train_ds = load_dataset_from_dir(TRAIN_DIR, validation_split=0.2, subset="training")
    val_ds = load_dataset_from_dir(TRAIN_DIR, validation_split=0.2, subset="validation")

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=default_callbacks())

    # Version the old model before overwriting, so you can roll back / show provenance
    if os.path.exists(MODEL_PATH):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.rename(MODEL_PATH, os.path.join(MODEL_DIR, f"beans_model_{stamp}.keras"))
    save_model(model)

    result = {
        "status": "success",
        "epochs_run": len(hist.history["loss"]),
        "best_val_accuracy": float(max(hist.history["val_accuracy"])),
        "final_train_accuracy": float(hist.history["accuracy"][-1]),
        "duration_seconds": round(time.time() - started, 1),
        "model_path": MODEL_PATH,
    }

    # Re-evaluate on the untouched test set so the UI can show fresh production metrics
    if os.path.isdir(TEST_DIR):
        try:
            test_ds = load_dataset_from_dir(TEST_DIR, shuffle=False)
            result["test_metrics"] = evaluate(model, test_ds)
        except Exception as exc:  # noqa: BLE001
            result["test_metrics_error"] = str(exc)

    return result
