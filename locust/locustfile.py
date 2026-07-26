"""
locustfile.py — flood-request simulation against the Bean Leaf Disease API.

Usage
-----
Web UI:
    locust -f locust/locustfile.py --host http://localhost:8000

Headless (what you'll use to record results for the README):
    locust -f locust/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 2m \
           --headless --csv results/1container_50users

Put a few sample .jpg leaf images in locust/sample_images/ first.
"""

import os
import random
import glob

from locust import HttpUser, task, between, events

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
_images = []


@events.test_start.add_listener
def load_samples(environment, **kwargs):
    """Read the sample images into memory once, before the flood starts,
    so disk I/O doesn't pollute the latency measurements."""
    global _images
    paths = glob.glob(os.path.join(SAMPLE_DIR, "*.jpg")) + glob.glob(
        os.path.join(SAMPLE_DIR, "*.png")
    )
    if not paths:
        raise RuntimeError(
            f"No sample images found in {SAMPLE_DIR}. "
            "Add a few leaf .jpg files before running the load test."
        )
    for p in paths:
        with open(p, "rb") as f:
            _images.append((os.path.basename(p), f.read()))
    print(f"[locust] loaded {len(_images)} sample images")


class PredictUser(HttpUser):
    """Simulates a farmer hitting the classifier from the field."""

    wait_time = between(0.5, 2.0)   # think-time between requests

    @task(10)
    def predict(self):
        name, data = random.choice(_images)
        with self.client.post(
            "/predict",
            files={"file": (name, data, "image/jpeg")},
            name="POST /predict",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "class" not in body:
                    resp.failure("no class in response")
                else:
                    resp.success()
            else:
                resp.failure(f"status {resp.status_code}")

    @task(2)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def data_stats(self):
        self.client.get("/data-stats", name="GET /data-stats")


class HeavyPredictUser(HttpUser):
    """No think-time — maximum pressure. Use with `--class-picker` or on its own
    to find the true saturation point of the service."""

    wait_time = between(0, 0)

    @task
    def predict(self):
        name, data = random.choice(_images)
        self.client.post(
            "/predict", files={"file": (name, data, "image/jpeg")}, name="POST /predict [heavy]"
        )
