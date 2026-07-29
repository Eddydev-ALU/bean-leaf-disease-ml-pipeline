# Bean Leaf Disease Classification, End-to-End ML Pipeline

Classifies bean leaf images into **Angular Leaf Spot**, **Bean Rust**, or **Healthy** using a
MobileNetV2 transfer-learning model, served through a FastAPI + Streamlit stack, containerised with
Docker, load-balanced with nginx, and load-tested with Locust.

**Dataset:** [AI-Lab-Makerere/beans (iBean)](https://huggingface.co/datasets/AI-Lab-Makerere/beans), 
1,295 smartphone photographs of bean leaves collected in the field across districts of Uganda by the
Makerere AI Lab in collaboration with NaCRRI.

---

## Links

| Item | Link |
|---|---|
| **Video demo** | [Bean-Leaf-ML-Pipeline-Demo-Video](https://youtu.be/wetNIEw9FKg) |
| **Live UI** | [bean-leaf-disease-ui.onrender.com](https://bean-leaf-disease-ui.onrender.com/) |
| **Live API docs (Swagger)** | [bean-leaf-disease-api.onrender.com/docs](https://bean-leaf-disease-api.onrender.com/docs) |
| **Training notebook** | [`notebook/bean_disease_classification.ipynb`](notebook/bean_disease_classification.ipynb) |
| **Model file** | [`models/beans_model.keras`](models/) (also exported as `.h5`) |

---

## Project description

Common bean is a staple protein crop for smallholder farmers across East Africa, nowhere more than Rwanda, which has the **highest bean consumption per capita of any country in the world** at 30.8 kg
per person per year, ahead of El Salvador and Tanzania
([Helgi Library, bean consumption per capita](https://www.helgilibrary.com/indicators/bean-consumption-per-capita/)).
Angular Leaf Spot and Bean Rust are two of its most damaging diseases, and both are diagnosable from a
single leaf photograph, but only if an agronomist is available. This project puts that diagnosis in a
phone browser: a farmer uploads a leaf photo and receives a classification in under 100 ms.

The system implements the complete ML lifecycle:

| Stage | Implementation |
|---|---|
| **Data acquisition** | iBean pulled directly from the Hugging Face Hub via `pandas.read_parquet`, reproducible, no manual downloads |
| **Data processing** | Decode parquet bytes → RGB → resize 224×224 → augment (flip / rotate / zoom / contrast) → MobileNetV2 preprocessing |
| **Model creation** | MobileNetV2 backbone (ImageNet), two-stage training: frozen-backbone head training, then partial fine-tuning |
| **Model testing** | 5 metrics + per-class report + confusion matrix + error analysis + threshold calibration on a held-out 128-image test split |
| **Retraining** | Users bulk-upload labelled images; a threshold trigger fires; the **existing custom model** is loaded and fine-tuned further; old versions are timestamped and kept |
| **API** | FastAPI, 8 endpoints, auto-generated Swagger docs |
| **UI** | Streamlit, uptime monitor, live data insights, prediction, bulk upload, retrain button |
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

- **Pretrained model**: MobileNetV2 with ImageNet weights as the backbone
- **Regularization**: Dropout (0.3, two layers), L2 weight decay (1e-4) on the dense head, plus data augmentation (horizontal flip, ±15% rotation, ±15% zoom, ±10% contrast)
- **Optimizer**: Adam, `1e-3` for head training, `3e-5` for fine-tuning
- **Early stopping**: on `val_accuracy` with `restore_best_weights=True`
- **LR scheduling**: `ReduceLROnPlateau`, factor 0.3
- **Hyperparameter search**: three fine-tuning configurations tested and compared

### What the numbers actually say

Three fine-tuning configurations were evaluated:

| Unfrozen layers | Fine-tune LR | Test accuracy | Test F1 | Test ROC-AUC | angular recall |
|---|---|---|---|---|---|
| 30 | `1e-5` | 0.8438 | 0.8405 | 0.9627 | 0.81 |
| 60 | `1e-4` | 0.7500 | 0.7352 | 0.9555 | 0.47 |
| **40** | **`3e-5`** | **0.8281** | **0.8245** | **0.9596** | **0.67** |

**ROC-AUC stayed within 0.9555–0.9627 across every configuration**, including the one that lost 9
percentage points of accuracy. The learned representation is stable and strong; only the argmax
decision boundary moves. Features fine, calibration fragile, that distinction drove the decision to
apply threshold calibration rather than simply train longer.

The validation sanity check also showed fine-tuning *reducing* validation accuracy (0.8120 → 0.7970),
with early stopping cutting it off at epoch 11 of 30. With only 1,034 training images, re-tuning 40
backbone layers overfits. This is documented honestly in the notebook rather than hidden.

### The error that matters operationally

`healthy` recall is **1.00 across all three configurations**, the model never misses a healthy leaf.
But `healthy` precision is only 0.82, meaning **9 of 86 diseased leaves (10.5%) are classified as
healthy**. In the field that is false reassurance: telling a farmer an infected plant is fine. A false
alarm costs one inspection; a missed infection can cost a harvest.

Of 22 total test errors, **13 come from the Angular ↔ Rust boundary**, more than the other two class
pairs combined. Both diseases produce brown-ish spotting and differ mainly in lesion *shape* (angular
and vein-bounded vs small round pustules). **Actionable conclusion:** further data collection should
prioritise close-up, well-lit images of these two classes, not more healthy leaves.

---

## Repository structure

```
bean-leaf-disease/
├── README.md
├── prepare_data.py              
├── run_load_tests.sh            
├── docker-compose.yml
├── requirements-api.txt / -ui.txt / -dev.txt
│
├── notebook/
│   └── bean_disease_classification.ipynb
│
├── src/
│   ├── preprocessing.py         
│   ├── model.py                 
│   └── prediction.py            
│
├── app/
│   ├── api.py                   
│   └── ui.py                    
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── nginx.conf               
│
├── locust/
│   ├── locustfile.py
│   └── sample_images/
│
├── data/
│   ├── train/                   
│   └── test/                    
│
└── models/
    ├── beans_model.keras        
    ├── beans_model.h5           
    ├── class_names.json
    ├── metrics.json             
    └── insights.json            
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
containers is the only variable in the load test, this keeps the benchmark honest.

### Scale horizontally

```bash
docker compose up -d --scale api=3
docker compose ps
```

nginx picks up new replicas automatically via Docker's embedded DNS (`resolve` + 10s TTL). Every
`/predict` response includes a `served_by` field showing which container handled it, useful for
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

### Example: prediction

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

### Example: bulk upload then retrain

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

1. **Upload**: the UI's *Upload & Retrain* page accepts either multiple images with a chosen class
   label, or a `.zip` whose top-level folders are class names.
2. **Save**: `src/preprocessing.py:save_uploaded_image()` writes each file to
   `data/train/<class>/upload_NNNNN.jpg`. Because `./data` is bind-mounted, uploads survive container
   restarts.
3. **Preprocess**: `load_dataset_from_dir()` rebuilds the `tf.data` pipeline over the combined
   original + uploaded images, applying the identical resize → augment → MobileNetV2 preprocessing
   chain used in training.
4. **Trigger**: `should_retrain()` fires when ≥ `RETRAIN_THRESHOLD` (default 20) new images have
   accumulated, or when tracked accuracy falls below `ACC_FLOOR` (default 0.80). The UI also exposes a
   *force* override.
5. **Retrain**: `src/model.py:retrain()` loads `models/beans_model.keras`, **our own trained model,
   used as the pretrained starting point**, and fine-tunes it at `1e-5` with early stopping on
   `val_accuracy`.
6. **Version**: the previous model is renamed `beans_model_<timestamp>.keras` before the new one is
   written, so you can always roll back.
7. **Hot reload**: `src/prediction.py:reload_model()` clears the cached model so the API serves the
   new weights immediately, with no restart.
8. **Re-evaluate**: the fresh model is scored against the untouched test split and `metrics.json` is
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

**Test configuration:** 50 concurrent users · spawn rate 10/s · 2 minutes per run · 1 CPU per
container · `POST /predict` weighted 10×, `GET /health` 2×, `GET /data-stats` 1×

| Containers | Users | Total requests | Failures | RPS | Median (ms) | 95th %ile (ms) | 99th %ile (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 543 | 0 (0.00%) | 4.58 | 9900 | 14000 | 23000 | 44050 |
| 2 | 50 | 1964 | 2 (0.10%) | 16.38 | 2500 | 3500 | 5700 | 7361 |
| 3 | 50 | 2962 | 0 (0.00%) | 24.73 | 1500 | 2400 | 3000 | 6367 |

With a single container, median latency was **9,900 ms** and the 95th percentile reached **14,000
ms**, as every request queued behind one uvicorn worker doing synchronous CPU inference. Scaling to
two containers cut both by roughly **75%** (median to 2,500 ms, P95 to 3,500 ms) and raised throughput
from 4.58 to 16.38 RPS, the classic signature of relieving a queueing bottleneck rather than making
the model itself faster. The third container pushed total throughput to 24.73 RPS, but per-container
throughput plateaued at **~8.2 requests/s/replica** (up from 4.58 at n=1, essentially flat between
n=2 and n=3), with P95 improving only another ~31% instead of repeating the earlier ~75% drop. That
plateau is expected: TensorFlow CPU inference is compute-bound, so once queueing delay is mostly
gone, additional replicas start competing for the same finite physical cores instead of adding real
parallel capacity.

Raw Locust CSVs are committed under `results/` as evidence.

---

## Deployment

Deployed on [Render](https://render.com) as two separate Docker-based Web Services built from this
repo, `bean-api` and `bean-ui`.

### Steps

1. Push to GitHub. The model (`models/beans_model.keras`, ~24 MB) and the full training/test dataset
   (`data/train`, `data/test`, ~67 MB) are both committed directly, no Git LFS needed, so the API
   image is self-contained and doesn't depend on any bind-mounted volume at runtime.
2. **API service** (`bean-api`):
   - **New +** → **Web Service** → connect the repo
   - Runtime: **Docker**, Dockerfile Path: `docker/Dockerfile.api`
   - Instance Type: **Free**
   - After the first deploy, go to **Settings → Health Checks** and set the Health Check Path to
     `/health`
   - No environment variables are required, `MODEL_DIR`/`DATA_DIR` are baked into the image, and
     Render injects `PORT` automatically (both Dockerfiles bind to `${PORT}` with a local fallback,
     rather than a hardcoded port, since Render's `EXPOSE`-based port auto-detection isn't guaranteed)
3. **UI service** (`bean-ui`):
   - Same repo, Dockerfile Path: `docker/Dockerfile.ui`, Instance Type: **Free**
   - Health Check Path: `/_stcore/health`
   - Environment variable: `API_URL` = the API service's public URL from step 2
4. Paste both resulting URLs into the Links table at the top of this README.

### Known limitations of Render's Free tier, acknowledge these on camera

- **512 MB RAM** is tight for TensorFlow; the API can be OOM-killed under sustained load. A paid
  **Standard** plan (2 GB) removes this risk entirely if budget allows, Free was used here deliberately
  to keep the deployment cost-free for the demo.
- **Cold starts**: a Free instance spins down after ~15 minutes idle and takes 30–60s to wake on the
  next request. Hit `/health` a minute before recording to warm it up.
- **Ephemeral filesystem**: the baseline dataset ships baked into the image, so the hosted UI shows
  real training counts and retraining has real data to start from, but anything **uploaded after
  deploy** (new images, a freshly retrained model) is lost on the next restart or redeploy. Demonstrate
  the full persistent upload → retrain cycle on the local Docker Compose stack instead, where `./data`
  and `./models` are bind-mounted to the host.

### Evaluating the model in production

The deployed system evaluates itself, which is what "demonstrate the evaluation process in
production" asks for:

- `GET /metrics` serves the current model's accuracy, precision, recall, F1, ROC-AUC and confusion
  matrix, rendered on the UI's **Model Performance** page.
- After every retrain, `src/model.py:retrain()` re-scores the new model against the untouched test
  split and rewrites `metrics.json`, so the dashboard always reflects the model actually being served.
- `GET /health` reports uptime and model-load status; the UI sidebar polls it continuously.
- `should_retrain()` monitors accuracy against `ACC_FLOOR` and flags when the served model has decayed
  below acceptable quality.

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
