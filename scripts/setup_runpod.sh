#!/bin/bash
# Setup torchtitan on RunPod: detect GPU architecture, install matching PyTorch nightly, and deps.
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

echo "=== RunPod Setup ==="

# --- Detect GPU compute capability ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
# compute_cap comes as e.g. "12.0" — extract major version
CC_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

echo "  GPU: $GPU_NAME (compute capability $COMPUTE_CAP)"

# --- Pick the right PyTorch CUDA index based on compute capability ---
# sm_120 (Blackwell) needs cu128+, sm_90 (Hopper) needs cu126+, older works with cu124
if [ "$CC_MAJOR" -ge 12 ]; then
    CUDA_INDEX="cu128"
elif [ "$CC_MAJOR" -ge 9 ]; then
    CUDA_INDEX="cu126"
else
    CUDA_INDEX="cu124"
fi

echo "  Selected PyTorch index: $CUDA_INDEX"

# --- Check if current PyTorch is already correct ---
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
NEEDS_INSTALL=true

if [[ "$CURRENT_TORCH" == *"$CUDA_INDEX"* ]] && [[ "$CURRENT_TORCH" == *dev* ]]; then
    # Verify the GPU is actually usable with this build
    if python -c "import torch; torch.cuda.get_device_name(0)" 2>/dev/null | grep -qi "not compatible"; then
        echo "  Current PyTorch ($CURRENT_TORCH) incompatible with GPU, reinstalling..."
    else
        echo "  PyTorch $CURRENT_TORCH already installed with $CUDA_INDEX"
        NEEDS_INSTALL=false
    fi
else
    echo "  Current PyTorch: $CURRENT_TORCH (need nightly+$CUDA_INDEX)"
fi

if $NEEDS_INSTALL; then
    echo "  Installing PyTorch nightly ($CUDA_INDEX)..."
    pip install --pre --force-reinstall torch \
        --index-url "https://download.pytorch.org/whl/nightly/$CUDA_INDEX" \
        2>&1 | tail -3
fi

# --- Install test dependencies ---
echo ""
echo "=== Installing dependencies ==="
pip install expecttest -q 2>/dev/null
pip install -e "$REPO_DIR" -q 2>&1 | tail -1
pip install -r "$REPO_DIR/requirements.txt" -q 2>&1 | tail -1
if [ -f "$REPO_DIR/requirements-dev.txt" ]; then
    pip install -r "$REPO_DIR/requirements-dev.txt" -q 2>&1 | tail -1
fi

# --- Verify ---
echo ""
echo "=== Verification ==="
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
GPU_USABLE=$(python -c "
import torch
try:
    torch.zeros(1, device='cuda')
    print('yes')
except Exception as e:
    print(f'no ({e})')
")

echo "  PyTorch:    $TORCH_VER"
echo "  CUDA:       $CUDA_VER"
echo "  GPUs:       $GPU_COUNT × $GPU_NAME"
echo "  GPU usable: $GPU_USABLE"

if [ "$GPU_USABLE" != "yes" ]; then
    echo ""
    echo "[ERROR] GPU not usable with current PyTorch build."
    echo "  Try: CUDA_INDEX=cu126 bash scripts/setup_runpod.sh"
    exit 1
fi

echo ""
echo "[OK] Setup complete. Run tests with:"
echo "  bash scripts/test_mixtral_runpod.sh"
