#!/usr/bin/env bash
# Run the local Qwen2.5-0.5B experiment on circle packing.
# Requires llama.cpp server running at http://localhost:8080
#
# Usage:
#   bash experiments/circle_packing_local_vs_cloud/run_experiment.sh
#
# Prerequisites:
#   1. Start llama.cpp: ./llama-server -m /path/to/qwen2.5-0.5b.gguf --port 8080
#   (Python env is handled automatically via .venv in the repo root)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use the repo's own .venv so activation is never required
PYTHON="$REPO_DIR/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: .venv not found at $REPO_DIR/.venv"
    echo "Run:  uv venv --python 3.13 --clear .venv && uv pip install -e '.[dev]'"
    exit 1
fi
OUTPUT_DIR="$SCRIPT_DIR/results/local"
CONFIG="$SCRIPT_DIR/config_local.yaml"
EXPERIMENTS_FILE="$REPO_DIR/experiments/EXPERIMENTS.md"

echo "=== Circle Packing: Qwen2.5-0.5B Local Model Experiment ==="
echo "Date: $(date)"
echo "Repo:   $REPO_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# ── 1. Check llama.cpp server ─────────────────────────────────────────────────
echo "[1/3] Checking llama.cpp server at http://localhost:8080 ..."
if ! curl -sf "http://localhost:8080/v1/models" \
        -H "Authorization: Bearer local" \
        -o /dev/null 2>&1; then
    echo ""
    echo "ERROR: llama.cpp server not responding at http://localhost:8080"
    echo ""
    echo "Start it first, e.g.:"
    echo "  ./llama-server -m /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080"
    echo ""
    exit 1
fi

# Auto-detect model name from server (used to override config at runtime)
MODEL_NAME=$(curl -sf "http://localhost:8080/v1/models" \
    -H "Authorization: Bearer local" \
    | "$PYTHON" -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d['data'][0]['id'])
except Exception:
    print('Qwen2.5-0.5B')
" 2>/dev/null || echo "Qwen2.5-0.5B")

echo "Server OK. Model: $MODEL_NAME"
echo ""

mkdir -p "$OUTPUT_DIR"

# ── 2. Run evolution ──────────────────────────────────────────────────────────
echo "[2/3] Running OpenEvolve for 30 iterations ..."
echo "      Config:  $CONFIG"
echo "      Results: $OUTPUT_DIR"
echo ""

START_EPOCH=$(date +%s)

"$PYTHON" "$REPO_DIR/openevolve-run.py" \
    "$REPO_DIR/examples/circle_packing/initial_program.py" \
    "$REPO_DIR/examples/circle_packing/evaluator.py" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR" \
    --iterations 30 \
    --primary-model "$MODEL_NAME"

END_EPOCH=$(date +%s)
WALL_TIME=$(( END_EPOCH - START_EPOCH ))
MINS=$(( WALL_TIME / 60 ))
SECS=$(( WALL_TIME % 60 ))

echo ""
echo "Evolution complete. Wall time: ${WALL_TIME}s (${MINS}m ${SECS}s)"
echo ""

# ── 3. Analyze and write results ──────────────────────────────────────────────
echo "[3/3] Analyzing results → $EXPERIMENTS_FILE"
echo ""

"$PYTHON" "$SCRIPT_DIR/analyze_results.py" \
    --local    "$OUTPUT_DIR" \
    --output   "$EXPERIMENTS_FILE" \
    --wall-time "$WALL_TIME" \
    --model    "$MODEL_NAME" \
    --iterations 30

echo ""
echo "Done. Open experiments/EXPERIMENTS.md to read the verdict."
