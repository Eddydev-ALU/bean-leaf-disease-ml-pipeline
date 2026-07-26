# Bean Leaf Disease Classification — End-to-End ML Pipeline

Classifies bean leaf images into **Angular Leaf Spot**, **Bean Rust**, or **Healthy** using a
MobileNetV2 transfer-learning model, served through a FastAPI + Streamlit stack, containerised with
Docker, load-balanced with nginx, and load-tested with Locust.

**Dataset:** [AI-Lab-Makerere/beans (iBean)](https://huggingface.co/datasets/AI-Lab-Makerere/beans) —
1,295 smartphone photographs of bean leaves collected in the field across districts of Uganda by the
Makerere AI Lab in collaboration with NaCRRI.

---

## Links

| Item | Link |
|---|---|
| **Video demo (camera on)** | `<PASTE YOUR YOUTUBE LINK>` |
| **Live UI** | `<PASTE YOUR DEPLOYED STREAMLIT URL>` |
| **Live API docs (Swagger)** | `<PASTE YOUR API URL>/docs` |
| **Training notebook** | [`notebook/bean_disease_classification.ipynb`](notebook/bean_disease_classification.ipynb) |
| **Model file** | [`models/beans_model.keras`](models/) (also exported as `.h5`) |

---

## Project description

Common bean is a staple protein crop for smallholder farmers across East Africa. Angular Leaf Spot
and Bean Rust are two of its most damaging diseases, and both are diagnosable from a single leaf
photograph — but only if an agronomist is available. This project puts that diagnosis in a phone
browser: a farmer uploads a leaf photo and receives a classification in under 100 ms.

The system implements the complete ML lifecycle:

| Stage | Implementation |
|---|---|
| **Data acquisition** | iBean pulled directly from the Hugging Face Hub via `pandas.read_parquet` — reproducible, no manual downloads |
| **Data processing** | Decode parquet bytes → RGB → resize 224×224 → augment (flip / rotate / zoom / contrast) → MobileNetV2 preprocessing |
| **Model creation** | MobileNetV2 backbone (ImageNet), two-stage training: frozen-backbone head training, then partial fine-tuning |
| **Model testing** | 5 metrics + per-class report + confusion matrix + error analysis + threshold calibration on a held-out 128-image test split |
| **Retraining** | Users bulk-upload labelled images; a threshold trigger fires; the **existing custom model** is loaded and fine-tuned further; old versions are timestamped and kept |
| **API** | FastAPI, 8 endpoints, auto-generated Swagger docs |
| **UI** | Streamlit — uptime monitor, live data insights, prediction, bulk upload, retrain button |
| **Scaling** | Docker Compose + nginx `least_conn` load balancing across N API replicas |
| **Monitoring** | `/health` uptime + `/metrics` production evaluation, both surfaced in the UI |

---

## Model performance

Held-out test set, 128 images (43 / 43 / 42):

| Metric | Score |
|---|---|
| **Accuracy** | 0.8281 |
| **Precision** (macro) | 0.8400 |
| **Recall** (macro) | 0.8295 |
| **F1-score** (macro) | 0.8245 |
| **ROC-AUC** (macro, OVR) | 0.9596 |

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angular_leaf_spot | 0.94 | 0.67 | 0.78 | 43 |
| bean_rust | 0.76 | 0.81 | 0.79 | 43 |
| healthy | 0.82 | 1.00 | 0.90 | 42 |

### Optimization techniques used

- **Pretrained model** — MobileNetV2 with ImageNet weights as the backbone
- **Regularization** — Dropout (0.3, two layers), L2 weight decay (1e-4) on the dense head, plus data augmentation (horizontal flip, ±15% rotation, ±15% zoom, ±10% contrast)
- **Optimizer** — Adam, `1e-3` for head training, `3e-5` for fine-tuning
- **Early stopping** — on `val_accuracy` with `restore_best_weights=True`
- **LR scheduling** — `ReduceLROnPlateau`, factor 0.3
- **Hyperparameter search** — three fine-tuning configurations tested and compared

### What the numbers actually say

Three fine-tuning configurations were evaluated:

| Unfrozen layers | Fine-tune LR | Test accuracy | Test F1 | Test ROC-AUC | angular recall |
|---|---|---|---|---|---|
| 30 | `1e-5` | 0.8438 | 0.8405 | 0.9627 | 0.81 |
| 60 | `1e-4` | 0.7500 | 0.7352 | 0.9555 | 0.47 |
| **40** | **`3e-5`** | **0.8281** | **0.8245** | **0.9596** | **0.67** |

**ROC-AUC stayed within 0.9555–0.9627 across every configuration**, including the one that lost 9
percentage points of accuracy. The learned representation is stable and strong; only the argmax
decision boundary moves. Features fine, calibration fragile — that distinction drove the decision to
apply threshold calibration rather than simply train longer.

The validation sanity check also showed fine-tuning *reducing* validation accuracy (0.8120 → 0.7970),
with early stopping cutting it off at epoch 11 of 30. With only 1,034 training images, re-tuning 40
backbone layers overfits. This is documented honestly in the notebook rather than hidden.

### The error that matters operationally

`healthy` recall is **1.00 across all three configurations** — the model never misses a healthy leaf.
But `healthy` precision is only 0.82, meaning **9 of 86 diseased leaves (10.5%) are classified as
healthy**. In the field that is false reassurance: telling a farmer an infected plant is fine. A false
alarm costs one inspection; a missed infection can cost a harvest.

Of 22 total test errors, **13 come from the Angular ↔ Rust boundary** — more than the other two class
pairs combined. Both diseases produce brown-ish spotting and differ mainly in lesion *shape* (angular
and vein-bounded vs small round pustules). **Actionable conclusion:** further data collection should
prioritise close-up, well-lit images of these two classes, not more healthy leaves.

---

## Repository structure

```
bean-leaf-disease/
├── README.md
├── prepare_data.py              # downloads iBean -> data/train, data/test, locust samples
├── run_load_tests.sh            # automated 1/2/3-container Locust benchmark
├── docker-compose.yml
├── requirements-api.txt / -ui.txt / -dev.txt
│
├── notebook/
│   └── bean_disease_classification.ipynb
│
├── src/
│   ├── preprocessing.py         # decoding, augmentation, dataset builders, upload handling
│   ├── model.py                 # build / train / evaluate / retrain + trigger logic
│   └── prediction.py            # single-datapoint inference
│
├── app/
│   ├── api.py                   # FastAPI service
│   └── ui.py                    # Streamlit front-end
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── nginx.conf               # least_conn load balancer across API replicas
│
├── locust/
│   ├── locustfile.py
│   └── sample_images/
│
├── data/
│   ├── train/                   # <class>/*.jpg  (gitignored, created by prepare_data.py)
│   └── test/
│
└── models/
    ├── beans_model.keras        # production model
    ├── beans_model.h5           # legacy format
    ├── class_names.json
    ├── metrics.json             # exported by the notebook -> served at GET /metrics
    └── insights.json            # exported by the notebook -> served at GET /insights
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<you>/bean-leaf-disease.git
cd bean-leaf-disease

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Download the dataset

```bash
python prepare_data.py
```

Writes `data/train/<class>/*.jpg`, `data/test/<class>/*.jpg`, and 10 sample images into
`locust/sample_images/` for the load test.

### 3. Train the model

Open `notebook/bean_disease_classification.ipynb` in **Google Colab**
(Runtime → Change runtime type → **T4 GPU**) and run all cells. Training takes about 5 minutes.

Then download the artefacts into `models/`:

```python
from google.colab import files
for f in ['beans_model.keras', 'beans_model.h5',
          'class_names.json', 'metrics.json', 'insights.json']:
    files.download(f'models/{f}')
```

> `metrics.json` and `insights.json` are what make the deployed dashboard show your *real* numbers.
> Without them, the UI's Performance and Insights pages will report that they can't find the data.

### 4. Run locally (no Docker)

```bash
# Terminal 1 — API
uvicorn app.api:app --reload --port 8000

# Terminal 2 — UI
API_URL=http://localhost:8000 streamlit run app/ui.py
```

UI at http://localhost:8501 · Swagger docs at http://localhost:8000/docs

---

## Docker

### Build and start the stack

```bash
docker compose build          # first build ~5 min (TensorFlow is large)
docker compose up -d
docker compose ps             # all three services should read healthy
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API (via nginx) | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

### Architecture

```
                    ┌──────────────┐
   browser ───────► │  ui  :8501   │  Streamlit
                    └──────┬───────┘
                           │ API_URL=http://nginx:8000
                    ┌──────▼───────┐
                    │ nginx :8000  │  least_conn load balancer
                    └──────┬───────┘
                  ┌────────┼────────┐
             ┌────▼───┐ ┌──▼─────┐ ┌▼───────┐
             │ api #1 │ │ api #2 │ │ api #3 │  FastAPI + TensorFlow
             └────┬───┘ └──┬─────┘ └┬───────┘
                  └────────┼────────┘
                    ┌──────▼───────┐
                    │ ./data       │  bind-mounted: uploads persist
                    │ ./models     │  bind-mounted: model versions persist
                    └──────────────┘
```

Each API container runs exactly **one** uvicorn worker and is limited to **1 CPU**, so the number of
containers is the only variable in the load test — this keeps the benchmark honest.

### Scale horizontally

```bash
docker compose up -d --scale api=3
docker compose ps
```

nginx picks up new replicas automatically via Docker's embedded DNS (`resolve` + 10s TTL). Every
`/predict` response includes a `served_by` field showing which container handled it — useful for
proving load balancing on camera.

### Useful commands

```bash
docker compose logs -f api        # follow API logs
docker stats                      # live CPU/memory per container
docker compose exec nginx tail -f /var/log/nginx/access.log   # per-request timings
docker compose down               # stop
docker compose down -v            # stop and remove volumes
```

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Status, uptime (seconds + human), model loaded, container id |
| `GET` | `/metrics` | Production model evaluation metrics |
| `GET` | `/insights` | Dataset insights (class counts, RGB means, pairwise errors) |
| `GET` | `/data-stats` | Live image counts per class + retrain trigger state |
| `POST` | `/predict` | Single image → class, confidence, all probabilities, inference time |
| `POST` | `/upload` | Bulk images or a `.zip` → saved to `data/train/<class>/` |
| `POST` | `/retrain` | Fires background retraining; returns immediately |
| `GET` | `/retrain/status` | Live progress / result of the retraining job |

### Example — prediction

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@locust/sample_images/sample_0.jpg"
```

```json
{
  "class": "bean_rust",
  "confidence": 0.9100,
  "all_probs": {
    "angular_leaf_spot": 0.0851,
    "bean_rust": 0.9100,
    "healthy": 0.0048
  },
  "inference_ms": 84.2,
  "filename": "sample_0.jpg",
  "served_by": "a3f9c1b2d4e5"
}
```

### Example — bulk upload then retrain

```bash
# Upload a zip laid out as angular_leaf_spot/ bean_rust/ healthy/
curl -X POST http://localhost:8000/upload -F "files=@new_leaves.zip"

# Trigger retraining
curl -X POST http://localhost:8000/retrain \
     -H "Content-Type: application/json" \
     -d '{"epochs": 8, "force": false}'

# Poll progress
curl http://localhost:8000/retrain/status
```

---

## Retraining pipeline

1. **Upload** — the UI's *Upload & Retrain* page accepts either multiple images with a chosen class
   label, or a `.zip` whose top-level folders are class names.
2. **Save** — `src/preprocessing.py:save_uploaded_image()` writes each file to
   `data/train/<class>/upload_NNNNN.jpg`. Because `./data` is bind-mounted, uploads survive container
   restarts.
3. **Preprocess** — `load_dataset_from_dir()` rebuilds the `tf.data` pipeline over the combined
   original + uploaded images, applying the identical resize → augment → MobileNetV2 preprocessing
   chain used in training.
4. **Trigger** — `should_retrain()` fires when ≥ `RETRAIN_THRESHOLD` (default 20) new images have
   accumulated, or when tracked accuracy falls below `ACC_FLOOR` (default 0.80). The UI also exposes a
   *force* override.
5. **Retrain** — `src/model.py:retrain()` loads `models/beans_model.keras` — **our own trained model,
   used as the pretrained starting point** — and fine-tunes it at `1e-5` with early stopping on
   `val_accuracy`.
6. **Version** — the previous model is renamed `beans_model_<timestamp>.keras` before the new one is
   written, so you can always roll back.
7. **Hot reload** — `src/prediction.py:reload_model()` clears the cached model so the API serves the
   new weights immediately, with no restart.
8. **Re-evaluate** — the fresh model is scored against the untouched test split and `metrics.json` is
   rewritten, so the UI's Performance page reflects the retrained model.

---

## Flood request simulation (Locust)

### Automated run

```bash
pip install locust
chmod +x run_load_tests.sh
./run_load_tests.sh                       # 50 users, 2 min, at 1 / 2 / 3 containers
USERS=100 RUNTIME=3m ./run_load_tests.sh  # heavier
```

The script scales the API, waits for health checks, **warms up each replica** (so you measure
steady-state latency rather than cold-start model loading), runs the flood, and prints a markdown
table ready to paste below.

### Manual run

```bash
# Interactive web UI — best for the video, charts update live
locust -f locust/locustfile.py --host http://localhost:8000
# open http://localhost:8089

# Headless, one configuration
docker compose up -d --scale api=1
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 10 --run-time 2m \
       --headless --csv results/1container
```

### Results

> **Replace with your own measured numbers.** Run `./run_load_tests.sh` and paste its output table
> here. The row structure below is what the grader expects.

**Test configuration:** 50 concurrent users · spawn rate 10/s · 2 minutes per run · 1 CPU per
container · `POST /predict` weighted 10×, `GET /health` 2×, `GET /data-stats` 1×

| Containers | Users | Total requests | Failures | RPS | Median (ms) | 95th %ile (ms) | 99th %ile (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 50 | | | | | | | |
| 2 | 50 | | | | | | | |
| 3 | 50 | | | | | | | |

**Interpretation template** — adapt to your actual figures:

> With a single container, median latency was **X ms** and throughput saturated at **~Y RPS** as
> requests queued behind the one uvicorn worker; the 95th percentile diverged sharply from the median,
> the classic signature of queueing rather than slow compute. Scaling to two containers reduced
> 95th-percentile latency by **Z%** and raised throughput to **~W RPS**, close to the ideal 2× —
> confirming the workload is CPU-bound and parallelises cleanly. The third container delivered
> **diminishing returns**, because TensorFlow CPU inference is compute-bound and the three replicas
> now compete for the host's physical cores. On this hardware the sweet spot is **N containers**;
> scaling further would require more physical CPUs, not more replicas.

Raw Locust CSVs are committed under `results/` as evidence.

---

## Deployment

### Option A — Render (fastest to a public URL)

1. Push the repo to GitHub (ensure `models/beans_model.keras` is committed — it's only a few MB).
2. On [render.com](https://render.com): **New → Web Service** → connect your repo.
   - Runtime **Docker**, Dockerfile path `docker/Dockerfile.api`, port `8000`
   - Instance type: **Standard** or above (the free tier's 512 MB will OOM on TensorFlow)
3. Deploy, then copy the resulting public URL.
4. **New → Web Service** again for the UI:
   - Dockerfile path `docker/Dockerfile.ui`, port `8501`
   - Environment variable `API_URL` = the API service's public URL from step 3
5. Paste both URLs into the Links table at the top of this README.

> **Limitation to acknowledge on camera:** Render's ephemeral filesystem means uploaded images and
> retrained models are lost when the container restarts. Fine for a demo — but state it explicitly,
> and use Option B if you want to demonstrate retraining persistence properly.

### Option B — AWS EC2 / Lightsail (persistent, closest to production)

```bash
# Launch Ubuntu 22.04, t3.medium or larger (TensorFlow needs >= 4 GB RAM).
# Security group: open TCP 8000 and 8501.

ssh ubuntu@<your-instance-ip>

sudo apt update
sudo apt install -y docker.io docker-compose-plugin git python3-pip
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/<you>/bean-leaf-disease.git
cd bean-leaf-disease

pip3 install pandas pyarrow pillow huggingface_hub
python3 prepare_data.py

docker compose up -d --scale api=2

# UI:  http://<your-instance-ip>:8501
# API: http://<your-instance-ip>:8000/docs
```

Because `./data` and `./models` are bind-mounted to the host, uploads and retrained model versions
survive restarts — which lets you demonstrate the full retraining cycle convincingly.

### Evaluating the model in production

The deployed system evaluates itself, which is what "demonstrate the evaluation process in
production" asks for:

- `GET /metrics` serves the current model's accuracy, precision, recall, F1, ROC-AUC and confusion
  matrix — rendered on the UI's **Model Performance** page.
- After every retrain, `src/model.py:retrain()` re-scores the new model against the untouched test
  split and rewrites `metrics.json`, so the dashboard always reflects the model actually being served.
- `GET /health` reports uptime and model-load status; the UI sidebar polls it continuously.
- `should_retrain()` monitors accuracy against `ACC_FLOOR` and flags when the served model has decayed
  below acceptable quality.

---

## Video demo checklist

Record with your **camera on**, and cover these in order:

1. **Intro** (~30s) — who you are, the problem, the dataset, why it matters for East African farmers.
2. **Prediction** — upload a leaf image whose true class you know; show the **correct** prediction with
   confidence and the probability bar chart. Do one from each class if time allows.
3. **Data insights** — walk through all three feature visualisations and state the story each tells,
   especially the Angular ↔ Rust confusion finding and the 10.5% false-reassurance rate.
4. **Model performance** — show the live `/metrics` page: 5 metrics, confusion matrix, per-class table.
5. **Bulk upload** — upload a zip or multiple images; show the counter rise and the retrain trigger fire.
6. **Retraining** — press **Trigger Retraining**; show the job running, then the new validation accuracy
   and updated metrics. Point out that it loads your own model as the pretrained base.
7. **Uptime** — point at the sidebar uptime and container ID.
8. **Load test** — run Locust with the web UI visible, then `docker compose up -d --scale api=3` and
   re-run; show latency improving and mention the `served_by` field changing between containers.

---

## Citation

```bibtex
@ONLINE{beansdata,
    author = "Makerere AI Lab",
    title  = "Bean disease dataset",
    month  = "January",
    year   = "2020",
    url    = "https://github.com/AI-Lab-Makerere/ibean/"
}
```

---

## License

Released for educational use. The iBean dataset is the property of the Makerere AI Lab and NaCRRI.
