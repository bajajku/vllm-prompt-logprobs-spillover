#!/usr/bin/env bash
set -euo pipefail

# End-to-end experiment runner (assumes vLLM server is already running via ./start_server.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$SCRIPT_DIR/.venv/bin/python}"
BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8025}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"
PLOTS_DIR="${PLOTS_DIR:-$SCRIPT_DIR/plots}"
SLO_P95_MS="${SLO_P95_MS:-500}"
RPS_GRID="${RPS_GRID:-4,8,12,16,20,24,28,32,36,40}"
SEEDS="${SEEDS:-101,202,303}"

echo "=== Step 1: Checking vLLM server health at ${BASE_URL} ==="
curl -sf "${BASE_URL}/health" >/dev/null
echo "Server version: $(curl -sf "${BASE_URL}/version")"

echo ""
echo "=== Step 2: Baseline capacity calibration (0% audit) ==="
"$PY" "$SCRIPT_DIR/benchmark.py" calibrate \
  --base-url "$BASE_URL" \
  --out-dir "$OUT_DIR" \
  --rps-grid "$RPS_GRID" \
  --slo-p95-ms "$SLO_P95_MS"

TARGET_RPS=$("$PY" -c "import json; print(json.load(open('$OUT_DIR/calibration/calibration_summary.json'))['recommended_75pct_rps'])")
echo "Using measured 75% baseline capacity target: ${TARGET_RPS} req/s"

echo ""
echo "=== Step 3: Main audit-fraction sweep (0%, 1%, 5%, 10% x 3 seeds) ==="
"$PY" "$SCRIPT_DIR/benchmark.py" sweep \
  --base-url "$BASE_URL" \
  --out-dir "$OUT_DIR" \
  --rps "$TARGET_RPS" \
  --audit-fracs "0.0,0.01,0.05,0.10" \
  --seeds "$SEEDS"

echo ""
echo "=== Step 4: Follow-up hunts (early vs late, serialization, load boundary) ==="
"$PY" "$SCRIPT_DIR/benchmark.py" hunts \
  --base-url "$BASE_URL" \
  --out-dir "$OUT_DIR" \
  --rps-75pct "$TARGET_RPS" \
  --seeds "$SEEDS"

echo ""
echo "=== Step 5: Generating Money Chart & Hunt Plots ==="
"$PY" "$SCRIPT_DIR/plot_results.py" \
  --results-dir "$OUT_DIR" \
  --out-dir "$PLOTS_DIR"

echo ""
echo "Done! Results in ${OUT_DIR} and plots in ${PLOTS_DIR}."
