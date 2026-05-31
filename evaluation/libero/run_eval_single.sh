#!/bin/bash
# =============================================================================
# Single-Suite LIBERO Evaluation Script (client only)
#
# Assumes the SimVLA policy server is already running (started separately in
# the simvla conda environment).
#
# Usage:
#   conda activate libero
#   ./run_eval_single.sh <task_suite> [num_trials] [num_workers] [host] [port]
#
# Examples:
#   ./run_eval_single.sh libero_goal
#   ./run_eval_single.sh libero_spatial 50 8 127.0.0.1 8102
#   ./run_eval_single.sh libero_10 20 4
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL="egl"

# =============================================================================
# Arguments
# =============================================================================
TASK_SUITE=${1:-"libero_goal"}
NUM_TRIALS=${2:-20}
NUM_WORKERS=${3:-4}
HOST=${4:-"127.0.0.1"}
PORT=${5:-8102}

echo "============================================"
echo " SimVLA Single-Suite LIBERO Evaluation"
echo "============================================"
echo "   Task Suite:   $TASK_SUITE"
echo "   Num Trials:   $NUM_TRIALS"
echo "   Num Workers:  $NUM_WORKERS"
echo "   Server:       ws://$HOST:$PORT"
echo "============================================"
echo ""

python "${SCRIPT_DIR}/evaluate_single_suite.py" \
    --host "$HOST" \
    --port "$PORT" \
    --task_suite "$TASK_SUITE" \
    --num_trials "$NUM_TRIALS" \
    --num_workers "$NUM_WORKERS" \
    --video_out "./eval_results"

echo ""
echo "Evaluation complete."
