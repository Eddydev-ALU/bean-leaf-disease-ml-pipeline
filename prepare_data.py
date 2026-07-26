"""
prepare_data.py — download the iBean dataset from Hugging Face and lay it out
as data/train/<class>/ and data/test/<class>/, plus grab a few sample images
for the Locust load test.

Run once locally (or in Colab) before building the Docker images:
    python prepare_data.py
"""
import os, io, shutil
import pandas as pd
from PIL import Image

BASE = "hf://datasets/AI-Lab-Makerere/beans/"
SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]


def row_to_pil(row):
    f = row["image"]
    return Image.open(io.BytesIO(f["bytes"])).convert("RGB") if isinstance(f, dict) else f.convert("RGB")


def save_split(df, split_dir):
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
    for i, (_, row) in enumerate(df.iterrows()):
        cls = CLASS_NAMES[int(row["labels"])]
        row_to_pil(row).save(os.path.join(split_dir, cls, f"{i:05d}.jpg"), "JPEG")
    print(f"  -> {len(df)} images written to {split_dir}")


if __name__ == "__main__":
    print("Downloading iBean from Hugging Face...")
    df_train = pd.read_parquet(BASE + SPLITS["train"])
    df_test = pd.read_parquet(BASE + SPLITS["test"])

    print("Writing train split...")
    save_split(df_train, "data/train")
    print("Writing test split...")
    save_split(df_test, "data/test")

    # a handful of images for the Locust flood test
    os.makedirs("locust/sample_images", exist_ok=True)
    for i in range(10):
        row_to_pil(df_test.iloc[i]).save(f"locust/sample_images/sample_{i}.jpg", "JPEG")
    print("Wrote 10 sample images to locust/sample_images/")
    print("Done.")
