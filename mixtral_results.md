# Mixtral Parallelism Benchmark Results

**Hardware:** 4x NVIDIA RTX PRO 4500 Blackwell (PCIe, 32 GiB each)
**Model:** Mixtral debugmodel (4 layers, 4 experts, dim=256, top_k=2)
**Config:** batch_size=4, seq_len=256, 10 training steps
**PyTorch:** 2.12.0.dev20260322+cu128
**Date:** 2026-03-23

## Summary

| Config | Final Loss | TPS | Memory | TFLOPS | Redistribute Warns |
|--------|-----------|-----|--------|--------|-------------------|
| [1/4] FSDP only (dp=4) | 7.0847 | **13,083** | 0.21 GiB | **0.38** | 0 |
| [2/4] FSDP+EP (dp=4, ep=2) | **6.9852** | 10,037 | 0.20 GiB | 0.29 | 0 |
| [3/4] FSDP+TP (dp=2, tp=2) | 6.9293 | 3,566 | 0.17 GiB | 0.10 | 4 |
| [4/4] FSDP+TP+EP+ETP (dp=2, tp=2, ep=2, etp=2) | 6.9473 | 3,031 | **0.15 GiB** | 0.09 | **8** |

## Detailed Metrics (rank 0)

### [1/4] FSDP Training (dp_shard=4)

Mesh dimensions: `[batch, loss, fsdp, efsdp]`

| Step | Loss | Grad Norm | Memory | TPS | TFLOPS | MFU |
|------|------|-----------|--------|-----|--------|-----|
| 1 | 8.2014 | 1.3531 | 0.19 GiB (0.61%) | 534 | 0.02 | 0.00% |
| 5 | 7.6927 | 1.6120 | 0.21 GiB (0.65%) | 13,049 | 0.38 | 0.12% |
| 10 | 7.0847 | 1.9953 | 0.21 GiB (0.65%) | 13,083 | 0.38 | 0.12% |

### [2/4] FSDP+EP Training (dp_shard=4, ep=2)

Mesh dimensions: `[batch, loss, fsdp, ep, efsdp]`

| Step | Loss | Grad Norm | Memory | TPS | TFLOPS | MFU |
|------|------|-----------|--------|-----|--------|-----|
| 1 | 8.1697 | 1.3705 | 0.19 GiB (0.59%) | 416 | 0.01 | 0.00% |
| 5 | 7.5753 | 1.6371 | 0.20 GiB (0.64%) | 10,010 | 0.29 | 0.09% |
| 10 | 6.9852 | 2.0208 | 0.20 GiB (0.64%) | 10,037 | 0.29 | 0.09% |

### [3/4] FSDP+TP Training (dp_shard=2, tp=2)

Mesh dimensions: `[batch, loss, fsdp, tp, efsdp]`

4 redistribute warnings: sequential all_reduce across `[fsdp, tp]` dimensions.

| Step | Loss | Grad Norm | Memory | TPS | TFLOPS | MFU |
|------|------|-----------|--------|-----|--------|-----|
| 1 | 8.0328 | 1.3644 | 0.16 GiB (0.50%) | 207 | 0.01 | 0.00% |
| 5 | 7.4592 | 1.7434 | 0.17 GiB (0.55%) | 3,525 | 0.10 | 0.03% |
| 10 | 6.9293 | 2.1046 | 0.17 GiB (0.55%) | 3,566 | 0.10 | 0.03% |

### [4/4] FSDP+TP+EP+ETP Training (dp_shard=2, tp=2, ep=2, etp=2)

Mesh dimensions: `[batch, loss, fsdp, tp, ep, etp, efsdp]`

8 redistribute warnings: sequential all_reduce across `[efsdp, ep, etp]` and `[fsdp, tp]` dimensions.

| Step | Loss | Grad Norm | Memory | TPS | TFLOPS | MFU |
|------|------|-----------|--------|-----|--------|-----|
| 1 | 8.0527 | 1.4133 | 0.14 GiB (0.44%) | 162 | 0.00 | 0.00% |
| 5 | 7.4386 | 1.7626 | 0.16 GiB (0.50%) | 2,964 | 0.09 | 0.03% |
| 10 | 6.9473 | 2.0884 | 0.15 GiB (0.49%) | 3,031 | 0.09 | 0.03% |

## Analysis

**Throughput:** FSDP-only is fastest at 13K TPS. Adding EP drops ~23% to 10K. Adding TP drops ~73% to 3.5K. Full parallelism (TP+EP+ETP) is slowest at 3K. TP has the highest communication overhead on this hardware since the 4 GPUs are connected via PCIe, not NVLink.

**Memory:** More parallelism dimensions = lower per-GPU memory. FSDP-only uses 0.21 GiB, full parallelism uses 0.15 GiB (29% reduction). TP shards weight matrices, EP shards experts — both reduce per-GPU footprint.

**Convergence:** All configs converge similarly (final loss 6.93–7.08). EP configs reach slightly lower loss, likely because expert sharding changes the effective routing/gradient dynamics on this tiny model.

**Communication overhead:** TP and ETP introduce redistribute warnings — the DTensor runtime needs multiple sequential all-reduce operations across mesh dimensions instead of single fused collectives. This is the main driver of the TPS drop.

**Bottom line:** For this small model on PCIe-connected GPUs, FSDP-only is optimal. TP/EP parallelism pays off at scale when model/expert sizes exceed single-GPU memory and GPUs have high-bandwidth interconnects (NVLink/NVSwitch).

## Deterministic Validation

Two identical runs with `--debug.seed=42 --debug.deterministic` produce **bit-wise identical** loss and grad_norm at every step (FSDP, dp_shard=4, seq_len=256).

| Step | Loss | Grad Norm | Match |
|------|------|-----------|-------|
| 1 | 8.15389 | 1.4766 | YES |
| 2 | 8.06480 | 1.4816 | YES |
| 3 | 7.93064 | 1.3212 | YES |
| 4 | 7.78407 | 1.5181 | YES |
| 5 | 7.55571 | 1.7636 | YES |
| 6 | 7.42311 | 1.9123 | YES |
| 7 | 7.27276 | 1.9784 | YES |
| 8 | 7.14593 | 2.0726 | YES |
| 9 | 7.04872 | 2.1212 | YES |
| 10 | 6.99158 | 2.1712 | YES |

All 10 steps match across both runs. The `torch.bincount` replacement for `torch.histc` is fully deterministic.
