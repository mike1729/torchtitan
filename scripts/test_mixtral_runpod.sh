#!/bin/bash
# Test Mixtral implementation on RunPod (4× RTX 2000 Ada)
# Usage: bash scripts/test_mixtral_runpod.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/torchtitan}"
BRANCH="${BRANCH:-mixtral-runpod}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

echo "============================================"
echo "Mixtral Test Suite — RunPod ($NGPU GPUs)"
echo "============================================"

# --- Setup ---
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH" 2>/dev/null || true
pip install -e . -q
echo "[OK] Setup complete"

check_loss() {
    local logfile="$1"
    python -c "
import re, sys
losses = []
with open('$logfile') as f:
    for line in f:
        m = re.search(r'loss:\s*([\d.]+)', line)
        if m:
            losses.append(float(m.group(1)))
if len(losses) < 2:
    print('  WARNING: Could not extract enough loss values')
    sys.exit(0)
print(f'  First loss: {losses[0]:.4f}, Last loss: {losses[-1]:.4f}')
if losses[-1] < losses[0]:
    print('  [OK] Loss decreased')
else:
    print('  [WARN] Loss did not decrease (may need more steps)')
"
}

# --- Unit tests ---
echo ""
echo "=== Unit Tests ==="
python -m pytest tests/unit_tests/test_mixtral.py -v --tb=short 2>&1 | tail -15
echo "[OK] Unit tests passed"

# --- GPU smoke test: single-GPU forward/backward ---
echo ""
echo "=== Single-GPU Forward/Backward ==="
python -c "
import torch
from torchtitan.models.mixtral import mixtral_configs

model = mixtral_configs['debugmodel'].build()
model.init_weights(buffer_device=torch.device('cuda'))
model = model.cuda()

tokens = torch.randint(0, 2048, (2, 64), device='cuda')
logits = model(tokens)
loss = torch.nn.functional.cross_entropy(logits.view(-1, 2048), tokens.view(-1))
loss.backward()

grad_count = sum(1 for p in model.parameters() if p.grad is not None)
total = sum(1 for _ in model.parameters())
print(f'  logits: {logits.shape}, loss: {loss.item():.4f}, grads: {grad_count}/{total}')
assert grad_count == total
print('  [OK]')
"

# --- State dict adapter round-trip ---
echo ""
echo "=== State Dict Adapter Round-Trip ==="
python -c "
import torch
from torchtitan.models.mixtral import mixtral_configs
from torchtitan.models.mixtral.state_dict_adapter import MixtralStateDictAdapter

model = mixtral_configs['debugmodel'].build()
model.init_weights(buffer_device=torch.device('cpu'))
sd = model.state_dict()

adapter = MixtralStateDictAdapter(mixtral_configs['debugmodel'], None)
hf_sd = adapter.to_hf(sd)
restored = adapter.from_hf(hf_sd)

# expert_bias buffers are not in HF format (expected)
missing = [k for k in sd if k not in restored and 'expert_bias' not in k]
shape_mismatch = [k for k in sd if k in restored and sd[k].shape != restored[k].shape]
value_mismatch = [k for k in sd if k in restored and not torch.equal(sd[k], restored[k])]

assert not missing, f'Missing keys: {missing}'
assert not shape_mismatch, f'Shape mismatches: {shape_mismatch}'
assert not value_mismatch, f'Value mismatches: {value_mismatch}'
print(f'  {len(sd)} TT keys -> {len(hf_sd)} HF keys -> {len(restored)} restored')
print('  [OK]')
"

# --- Test 1: FSDP only (4 GPU) ---
echo ""
echo "=== [1/4] FSDP Training (${NGPU} GPUs, 10 steps) ==="
torchrun --nproc-per-node="$NGPU" \
    -m torchtitan.train \
    --module mixtral --config mixtral_debugmodel \
    --training.steps 10 \
    --training.seq_len 256 \
    --training.local_batch_size 4 \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --metrics.log_freq 5 \
    2>&1 | tee /tmp/mixtral_fsdp.log
check_loss /tmp/mixtral_fsdp.log

# --- Test 2: FSDP + EP (4 GPU, EP=2) ---
echo ""
echo "=== [2/4] FSDP+EP Training (${NGPU} GPUs, EP=2, 10 steps) ==="
torchrun --nproc-per-node="$NGPU" \
    -m torchtitan.train \
    --module mixtral --config mixtral_debugmodel \
    --training.steps 10 \
    --training.seq_len 256 \
    --training.local_batch_size 4 \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --parallelism.expert_parallel_degree 2 \
    --metrics.log_freq 5 \
    2>&1 | tee /tmp/mixtral_ep.log
check_loss /tmp/mixtral_ep.log

# --- Test 3: FSDP + TP (4 GPU, TP=2, FSDP=2) ---
echo ""
echo "=== [3/4] FSDP+TP Training (${NGPU} GPUs, TP=2, FSDP=2, 10 steps) ==="
torchrun --nproc-per-node="$NGPU" \
    -m torchtitan.train \
    --module mixtral --config mixtral_debugmodel \
    --training.steps 10 \
    --training.seq_len 256 \
    --training.local_batch_size 4 \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2 \
    --metrics.log_freq 5 \
    2>&1 | tee /tmp/mixtral_tp.log
check_loss /tmp/mixtral_tp.log

# --- Test 4: FSDP + TP + EP + ETP (4 GPU) ---
echo ""
echo "=== [4/4] FSDP+TP+EP+ETP Training (${NGPU} GPUs, 10 steps) ==="
torchrun --nproc-per-node="$NGPU" \
    -m torchtitan.train \
    --module mixtral --config mixtral_debugmodel \
    --training.steps 10 \
    --training.seq_len 256 \
    --training.local_batch_size 4 \
    --parallelism.data_parallel_shard_degree 2 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2 \
    --parallelism.expert_tensor_parallel_degree 2 \
    --metrics.log_freq 5 \
    2>&1 | tee /tmp/mixtral_tp_ep.log
check_loss /tmp/mixtral_tp_ep.log

echo ""
echo "============================================"
echo "All tests complete!"
echo "============================================"
