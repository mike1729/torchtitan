#!/bin/bash
# Profile Mixtral parallelization configs on RunPod (4× GPU)
# Uses profilemodel (~3.6B params) for meaningful compute/communication ratios.
# Produces Chrome trace files in /tmp/mixtral_profiles/
#
# Usage: bash scripts/profile_mixtral.sh
# View:  Download trace JSON, open chrome://tracing or https://ui.perfetto.dev
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
PROFILE_DIR="/tmp/mixtral_profiles"
STEPS=10
CONFIG="mixtral_profilemodel"

# profile_freq=10, warmup=3, active=1 → captures step 10 after 3 warmup steps
# (wait=10-3-1=6 steps skipped, then 3 warmup, then 1 active capture)

echo "============================================"
echo "Mixtral Profiling — $NGPU GPUs, profilemodel (~3.6B params)"
echo "Traces will be saved to $PROFILE_DIR"
echo "============================================"

cd "$REPO_DIR"

run_profiled() {
    local name="$1"
    local trace_dir="$PROFILE_DIR/$name"
    shift
    echo ""
    echo "=== Profiling: $name ==="
    mkdir -p "$trace_dir"
    torchrun --nproc-per-node="$NGPU" \
        -m torchtitan.train \
        --module mixtral --config "$CONFIG" \
        --training.steps "$STEPS" \
        --metrics.log_freq 5 \
        --profiling.enable_profiling \
        --profiling.save_traces_folder "$trace_dir" \
        --profiling.profile_freq "$STEPS" \
        --profiling.profiler_warmup 3 \
        --profiling.profiler_active 1 \
        "$@" \
        2>&1 | tee "$trace_dir/training.log"
    echo "  Traces saved to $trace_dir"
}

# 1. FSDP only
run_profiled "fsdp" \
    --parallelism.data_parallel_shard_degree "$NGPU"

# 2. FSDP + EP
run_profiled "fsdp_ep" \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --parallelism.expert_parallel_degree 2

# 3. FSDP + TP
run_profiled "fsdp_tp" \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2

# 4. FSDP + TP + EP + ETP
run_profiled "fsdp_tp_ep_etp" \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2 \
    --parallelism.expert_tensor_parallel_degree 2

echo ""
echo "============================================"
echo "All profiles complete!"
echo ""
echo "Trace files:"
find "$PROFILE_DIR" -name "*.json" -type f | sort
echo ""
echo "To view: download the JSON files and open in"
echo "  chrome://tracing  or  https://ui.perfetto.dev"
echo "============================================"
