#!/usr/bin/env bash
#
# run_load_tests.sh — runs the identical Locust flood against 1, 2 and 3 API
# containers and prints a markdown table you can paste straight into the README.
#
# Usage:
#   chmod +x run_load_tests.sh
#   ./run_load_tests.sh              # 50 users, 2 minutes per run
#   USERS=100 RUNTIME=3m ./run_load_tests.sh
#
set -euo pipefail

USERS="${USERS:-50}"
SPAWN_RATE="${SPAWN_RATE:-10}"
RUNTIME="${RUNTIME:-2m}"
HOST="${HOST:-http://localhost:8000}"
OUTDIR="results"

mkdir -p "$OUTDIR"

if ! command -v locust >/dev/null 2>&1; then
  echo "locust not found. Install it with:  pip install locust"
  exit 1
fi

if ! ls locust/sample_images/*.jpg >/dev/null 2>&1; then
  echo "No sample images in locust/sample_images/. Run:  python prepare_data.py"
  exit 1
fi

for N in 1 2 3; do
  echo ""
  echo "=================================================================="
  echo " Scaling API to $N container(s)"
  echo "=================================================================="
  docker compose up -d --scale api="$N" >/dev/null

  # Wait for every replica to report healthy before flooding it — otherwise the
  # first run measures cold-start model loading instead of steady-state latency.
  echo "Waiting for replicas to become healthy..."
  for _ in $(seq 1 60); do
    if curl -fs "$HOST/health" >/dev/null 2>&1; then break; fi
    sleep 2
  done

  # Warm-up: load the model into memory on every replica so the measurement is fair.
  echo "Warming up (${N} replicas)..."
  for _ in $(seq 1 $((N * 5))); do
    curl -fs -X POST "$HOST/predict" \
         -F "file=@$(ls locust/sample_images/*.jpg | head -1)" >/dev/null 2>&1 || true
  done

  echo "Running Locust: $USERS users, spawn rate $SPAWN_RATE, duration $RUNTIME"
  locust -f locust/locustfile.py \
         --host "$HOST" \
         --users "$USERS" \
         --spawn-rate "$SPAWN_RATE" \
         --run-time "$RUNTIME" \
         --headless \
         --only-summary \
         --csv "$OUTDIR/${N}container"
done

echo ""
echo "=================================================================="
echo " RESULTS — paste this into your README"
echo "=================================================================="
python3 - <<'PYEOF'
import csv, os, glob

rows = []
for n in (1, 2, 3):
    path = f"results/{n}container_stats.csv"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("Name") == "Aggregated" or r.get("Type") == "":
                rows.append((n, r))
                break

def g(r, *keys, default="-"):
    for k in keys:
        if k in r and r[k] not in ("", "N/A"):
            return r[k]
    return default

print()
print("| Containers | Users | Total requests | Failures | RPS | Median (ms) | 95th %ile (ms) | 99th %ile (ms) | Max (ms) |")
print("|---|---|---|---|---|---|---|---|---|")
for n, r in rows:
    print(f"| {n} "
          f"| {os.environ.get('USERS','50')} "
          f"| {g(r,'Request Count')} "
          f"| {g(r,'Failure Count')} "
          f"| {float(g(r,'Requests/s',default=0)):.2f} "
          f"| {g(r,'Median Response Time','50%')} "
          f"| {g(r,'95%')} "
          f"| {g(r,'99%')} "
          f"| {g(r,'Max Response Time')} |")
print()
PYEOF

echo "Raw CSVs are in $OUTDIR/"
