# HSM Codebase - Code Review & Consolidation

## Summary

Reviewed and consolidated the HSM codebase. Removed duplicate implementations and unified on a single clean code path with full FAVOR+ PRF support.

---

## Codebase Consolidation ✅

### Files Deleted
- **`modules.py`** - Duplicate HSM implementation (used nn.Linear, no PRF support)
- **`generation.py`** - Duplicate generation code (replaced by inference.py)

### Files Migrated
- **`main.py`** - Now uses `GPT2WithHSM` from `model.py` and `generate()` from `inference.py`

### New Architecture (Single Code Path)

```
main.py / run_experiment.py
    └── model.py (GPT2WithHSM, GPT2WithSoftmax)
            └── hsm.py (HierarchicalSoftmaxHead with PRF)
                    └── prf.py (FAVOR+ implementation)
            └── inference.py (O(log V) generation)
            └── train.py (training loop)
            └── eval.py (metrics & evaluation)
    └── utils.py (Huffman tree, data loading)
```

---

## FAVOR+ PRF Implementation ✅

Fully implemented FAVOR+ Positive Random Features for sigmoid approximation, verified against [Google Research's implementation](https://github.com/google-research/google-research/tree/master/performer/fast_attention).

### Mathematical Basis

The FAVOR+ approach from the Performers paper (Choromanski et al., 2020) approximates exponential kernels:

```
exp(w^T h) ≈ φ(w)^T φ(h) * exp((||w||² + ||h||²) / 2 + max_w + max_h)
```

where the positive random feature map (with numerical stabilization) is:

```
φ(x) = [exp(x @ Ω^T - ||x||²/2 - max(...)) + ε] / sqrt(m)
```

- `Ω` is an orthogonal random matrix `[m, d]` (via QR decomposition)
- `m` = number of random features (default: 256)
- `d` = hidden dimension (768 for GPT-2)
- `ε` = small constant for numerical stability (1e-6)

### Key Functions in `prf.py`

| Function | Purpose |
|----------|---------|
| `orthogonal_random_features()` | Generate orthogonal Ω matrix via QR decomposition |
| `positive_random_features()` | Compute φ(x) with numerical stabilization |
| `prf_sigmoid_batched()` | Batched PRF sigmoid for training |
| `prf_sigmoid_single()` | Single-pair PRF sigmoid for inference |
| `PRFSigmoid` | Module wrapper with redraw support |

### Usage

```python
# Enable PRF during model creation
model = GPT2WithHSM(
    root=root,
    tokenizer=tokenizer,
    hidden_size=768,
    use_prf=True,           # Enable FAVOR+ approximation
    num_random_features=256, # Tradeoff: more = better approx, more compute
)

# Or in main.py, set the flag:
USE_PRF = True
```

---

## Bug Fixes Applied ✅

1. **Device mismatches** - All tensor creation now specifies device
2. **Buffer registration** - `redraw_random_features()` uses proper `register_buffer()`
3. **Numerical stability** - PRF implementation includes max-subtraction stabilization
4. **Deprecated APIs** - Replaced `F.sigmoid()` with `torch.sigmoid()`

---

## Current File Structure

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `eval.py` | Evaluation metrics & results |
| `hsm.py` | HierarchicalSoftmaxHead with PRF toggle |
| `inference.py` | O(log V) generation methods |
| `main.py` | Demo/testing entry point |
| `model.py` | GPT2WithHSM and GPT2WithSoftmax wrappers |
| `prf.py` | FAVOR+ PRF implementation |
| `run_experiment.py` | Full experiment runner with configs |
| `train.py` | Training loop and loss functions |
| `utils.py` | Huffman tree, tokenizer, data loading |

---

## Running the Code

```bash
# Quick demo/test
python -m src.main

# Full experiment with config
python -m src.run_experiment --config src/configs/run_1.yaml

# Run all experiments
python -m src.run_experiment --all
```

---

*Last updated: December 2024*
