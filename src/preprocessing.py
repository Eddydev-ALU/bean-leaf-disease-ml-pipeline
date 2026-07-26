"""
preprocessing.py
Image loading, decoding and dataset construction for the Bean Leaf Disease pipeline.
"""

import io
import os
import json
import zipfile
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

DATA_DIR = os.getenv("DATA_DIR", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


# --------------------------------------------------------------------------- #
# Single image helpers (used by the /predict endpoint)
# --------------------------------------------------------------------------- #
def bytes_to_pil(raw: bytes) -> Image.Image:
    """Decode raw uploaded bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_tensor(img: Image.Image) -> np.ndarray:
    """Resize + MobileNetV2-preprocess a PIL image into a (1,224,224,3) batch."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    return preprocess_input(arr)


def preprocess_upload(raw: bytes) -> np.ndarray:
    """Full path: raw bytes -> model-ready batch tensor."""
    return pil_to_tensor(bytes_to_pil(raw))


# --------------------------------------------------------------------------- #
# Augmentation (regularization) — training only
# --------------------------------------------------------------------------- #
def build_augmenter() -> tf.keras.Sequential:
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


# --------------------------------------------------------------------------- #
# Dataset construction from folders  (data/train/<class>/*.jpg)
# --------------------------------------------------------------------------- #
def load_dataset_from_dir(
    directory: str, shuffle: bool = True, validation_split: float = None, subset: str = None
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from an image directory laid out by class folder."""
    kwargs = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    if validation_split:
        kwargs.update(validation_split=validation_split, subset=subset)

    ds = tf.keras.utils.image_dataset_from_directory(directory, **kwargs)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def arrays_to_dataset(X: np.ndarray, y: np.ndarray, training: bool = False) -> tf.data.Dataset:
    """Build a dataset from in-memory arrays (used by retraining)."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(1000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda x, l: (preprocess_input(x), l), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# --------------------------------------------------------------------------- #
# Bulk upload handling (used by the /upload endpoint)
# --------------------------------------------------------------------------- #
def save_uploaded_image(raw: bytes, label: str, split: str = "train") -> str:
    """Save one uploaded image into data/<split>/<label>/ and return its path."""
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label '{label}'. Must be one of {CLASS_NAMES}")

    folder = os.path.join(DATA_DIR, split, label)
    os.makedirs(folder, exist_ok=True)

    img = bytes_to_pil(raw)
    idx = len(os.listdir(folder))
    path = os.path.join(folder, f"upload_{idx:05d}.jpg")
    img.save(path, "JPEG")
    return path


def extract_zip_uploads(raw_zip: bytes, split: str = "train") -> Tuple[int, List[str]]:
    """
    Accept a .zip whose top-level folders are class names, e.g.

        uploads.zip
        ├── angular_leaf_spot/  *.jpg
        ├── bean_rust/          *.jpg
        └── healthy/            *.jpg

    Saves every image into data/<split>/<class>/ and returns (count, errors).
    """
    saved, errors = 0, []
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        for member in zf.namelist():
            if member.endswith("/") or member.startswith("__MACOSX"):
                continue
            parts = member.split("/")
            label = next((p for p in parts if p in CLASS_NAMES), None)
            if label is None:
                errors.append(f"{member}: no recognised class folder")
                continue
            try:
                save_uploaded_image(zf.read(member), label, split=split)
                saved += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{member}: {exc}")
    return saved, errors


def count_images(directory: str = TRAIN_DIR) -> dict:
    """Return {class_name: n_images} for a split directory."""
    out = {}
    for cls in CLASS_NAMES:
        folder = os.path.join(directory, cls)
        out[cls] = len(os.listdir(folder)) if os.path.isdir(folder) else 0
    return out


def count_new_uploads(directory: str = TRAIN_DIR) -> int:
    """How many images arrived via upload (prefix 'upload_') — drives the retrain trigger."""
    n = 0
    for cls in CLASS_NAMES:
        folder = os.path.join(directory, cls)
        if os.path.isdir(folder):
            n += len([f for f in os.listdir(folder) if f.startswith("upload_")])
    return n


def load_class_names(path: str = "models/class_names.json") -> List[str]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return CLASS_NAMES
