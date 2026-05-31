#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script (parallel 4 task suites)
#
# Prerequisites:
#   1. Start the SimVLA policy server in the "simvla" conda environment first:
#        conda activate simvla
#        CUDA_VISIBLE_DEVICES=0 python ./evaluation/libero/serve_smolvlm_libero.py \
#            --checkpoint ./runs/simvla_libero_small/ckpt-180000 \
#            --norm_stats ./norm_stats/libero_norm.json \
#            --smolvlm_model ./pretrained/SmolVLM-500M-Instruct \
#            --port 8089 &
#
#   2. Then run THIS script in the "libero" conda environment:
#        conda activate libero
#        ./evaluation/libero/run_eval_all.sh
#
# =============================================================================

set -e

# =============================================================================
# Environment Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL="egl"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   MUJOCO_GL:   $MUJOCO_GL"
echo ""

# =============================================================================
# Arguments
# =============================================================================
PORT=${1:-8102}
NUM_TRIALS=${2:-20}
OUTPUT_PREFIX=${3:-"eval_simvla"}
GPUS=${4:-"4 5 6 7"}         # CUDA device IDs for each suite
NUM_PARALLEL=${5:-5}          # Rollout parallelism per suite (0 or 1 = sequential)

# Parse GPU list
read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: Need at least 4 GPUs, got ${#GPU_ARRAY[@]}"
    echo "   Usage: $0 <port> <num_trials> <output_prefix> \"<cuda_gpu1> <cuda_gpu2> <cuda_gpu3> <cuda_gpu4>\" [num_parallel]"
    exit 1
fi

GPU_SPATIAL=${GPU_ARRAY[0]}
GPU_OBJECT=${GPU_ARRAY[1]}
GPU_GOAL=${GPU_ARRAY[2]}
GPU_10=${GPU_ARRAY[3]}

# Output directory
OUTPUT_DIR="./eval_simvla_${PORT}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation..."
echo "   Server Port:   $PORT"
echo "   Num Trials:    $NUM_TRIALS"
echo "   Num Parallel:  $NUM_PARALLEL"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir:    $OUTPUT_DIR"
echo "   GPUs (CUDA):   spatial=$GPU_SPATIAL, object=$GPU_OBJECT, goal=$GPU_GOAL, 10=$GPU_10"
echo ""


# Run 4 task suites in parallel
echo "Launching 4 evaluation tasks..."

CUDA_VISIBLE_DEVICES=$GPU_SPATIAL python -u libero_client.py \
    --host 127.0.0.1 \
    --port $PORT \
    --client_type websocket \
    --task_suite libero_spatial \
    --num_trials $NUM_TRIALS \
    --num_parallel $NUM_PARALLEL \
    --video_out "$OUTPUT_DIR" > "${OUTPUT_PREFIX}_spatial.txt" 2>&1 &
PID_SPATIAL=$!
echo "   [PID $PID_SPATIAL] libero_spatial (CUDA $GPU_SPATIAL) -> ${OUTPUT_PREFIX}_spatial.txt"

CUDA_VISIBLE_DEVICES=$GPU_OBJECT python -u libero_client.py \
    --host 127.0.0.1 \
    --port $PORT \
    --client_type websocket \
    --task_suite libero_object \
    --num_trials $NUM_TRIALS \
    --num_parallel $NUM_PARALLEL \
    --video_out "$OUTPUT_DIR" > "${OUTPUT_PREFIX}_object.txt" 2>&1 &
PID_OBJECT=$!
echo "   [PID $PID_OBJECT] libero_object (CUDA $GPU_OBJECT) -> ${OUTPUT_PREFIX}_object.txt"

CUDA_VISIBLE_DEVICES=$GPU_GOAL python -u libero_client.py \
    --host 127.0.0.1 \
    --port $PORT \
    --client_type websocket \
    --task_suite libero_goal \
    --num_trials $NUM_TRIALS \
    --num_parallel $NUM_PARALLEL \
    --video_out "$OUTPUT_DIR" > "${OUTPUT_PREFIX}_goal.txt" 2>&1 &
PID_GOAL=$!
echo "   [PID $PID_GOAL] libero_goal (CUDA $GPU_GOAL) -> ${OUTPUT_PREFIX}_goal.txt"

CUDA_VISIBLE_DEVICES=$GPU_10 python -u libero_client.py \
    --host 127.0.0.1 \
    --port $PORT \
    --client_type websocket \
    --task_suite libero_10 \
    --num_trials $NUM_TRIALS \
    --num_parallel $NUM_PARALLEL \
    --video_out "$OUTPUT_DIR" > "${OUTPUT_PREFIX}_10.txt" 2>&1 &
PID_10=$!
echo "   [PID $PID_10] libero_10 (CUDA $GPU_10) -> ${OUTPUT_PREFIX}_10.txt"

echo ""
echo "Waiting for all evaluations to complete..."
echo "   Monitor progress with: tail -f ${OUTPUT_PREFIX}_*.txt"
echo ""

# Wait for all tasks
wait $PID_SPATIAL $PID_OBJECT $PID_GOAL $PID_10

echo ""
echo "All evaluations completed!"
echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    file="${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Success Rate|Average" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="
