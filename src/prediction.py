"""
prediction.py
Single-datapoint inference for the Bean Leaf Disease model.
"""

import os
import threading
from typing import Union

import numpy as np
from PIL import Image

from .preprocessing import pil_to_tensor, bytes_to_pil, load_class_names

MODEL_PATH = os.path.join(os.getenv("MODEL_DIR", "models"), "beans_model.keras")

_model = None
_lock = threading.Lock()


def get_model():
    """Lazily load the model once and cache it (thread-safe).
    Keeps container startup fast and avoids reloading on every request."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import tensorflow as tf

                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
                _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def reload_model():
    """Force a reload — call this right after retraining so the API serves the new weights."""
    global _model
    with _lock:
        _model = None
    return get_model()


def predict(img_input: Union[str, bytes, Image.Image]) -> dict:
    """
    Predict the class of a single bean leaf image.

    Accepts a file path, raw bytes (from an HTTP upload), or a PIL image.
    Returns {'class', 'confidence', 'all_probs'}.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, (bytes, bytearray)):
        img = bytes_to_pil(bytes(img_input))
    elif isinstance(img_input, Image.Image):
        img = img_input
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    batch = pil_to_tensor(img)
    probs = get_model().predict(batch, verbose=0)[0]

    class_names = load_class_names()
    idx = int(np.argmax(probs))
    return {
        "class": class_names[idx],
        "confidence": float(probs[idx]),
        "all_probs": {c: float(p) for c, p in zip(class_names, probs)},
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.prediction <image_path>")
        raise SystemExit(1)
    print(predict(sys.argv[1]))
