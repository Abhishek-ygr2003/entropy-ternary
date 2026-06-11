"""
Ternarization utilities.

Implements:
  - INT8 simulation
  - FC2-only ternarization
  - FC1 partial ternarization with three selection strategies:
      random, l2_norm, entropy_guided
  - Operation-count proxy metric
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Ternary core
# ──────────────────────────────────────────────────────────────

TAU = 0.3  # threshold fraction from the paper


def ternarize_weight_row(w: torch.Tensor) -> torch.Tensor:
    """
    Convert a single weight row to ternary representation.

    wt = +1  if w >  tau * max|W|
    wt = -1  if w < -tau * max|W|
    wt =  0  otherwise

    Effective weight: weff = wt * pnz  (pnz clipped to [0.1, 0.9])
    When wt == 0 => weff = 0
    """
    max_abs = w.abs().max().item()
    threshold = TAU * max_abs

    wt = torch.zeros_like(w)
    wt[w > threshold]  =  1.0
    wt[w < -threshold] = -1.0

    # Normalized magnitude, clipped to [0.1, 0.9]
    pnz = (w.abs() / (max_abs + 1e-8)).clamp(0.1, 0.9)
    pnz[wt == 0] = 0.0

    return wt * pnz


def ternarize_layer(weight: torch.Tensor, neuron_indices) -> torch.Tensor:
    """
    Ternarize selected rows (neurons) of a weight matrix.
    Returns a new weight tensor (does not modify in-place).
    """
    new_weight = weight.clone()
    for idx in neuron_indices:
        new_weight[idx] = ternarize_weight_row(weight[idx])
    return new_weight


# ──────────────────────────────────────────────────────────────
# INT8 simulation
# ──────────────────────────────────────────────────────────────

def simulate_int8(model) -> nn.Module:
    """
    Simulate INT8 quantization by quantizing all weights to 8-bit range.
    No structural change; just weight rounding.
    """
    model_int8 = copy.deepcopy(model)
    with torch.no_grad():
        for param in model_int8.parameters():
            w = param.data
            w_min, w_max = w.min(), w.max()
            scale = (w_max - w_min) / 255.0 + 1e-8
            quantized = torch.round((w - w_min) / scale).clamp(0, 255)
            param.data = quantized * scale + w_min
    return model_int8


# ──────────────────────────────────────────────────────────────
# Selection strategies
# ──────────────────────────────────────────────────────────────

def select_random(weight: torch.Tensor, fraction: float, seed: int = 42):
    """Uniform random neuron selection."""
    n_neurons = weight.size(0)
    n_select  = math.floor(fraction * n_neurons)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_neurons, size=n_select, replace=False).tolist()
    return sorted(indices)


def select_l2_norm(weight: torch.Tensor, fraction: float):
    """Select neurons in ascending order of L2 weight norm."""
    n_neurons = weight.size(0)
    n_select  = math.floor(fraction * n_neurons)
    norms = weight.norm(p=2, dim=1).cpu().numpy()
    indices = np.argsort(norms)[:n_select].tolist()
    return sorted(indices)


def _ternary_entropy(w_row: torch.Tensor) -> float:
    """Shannon entropy of the ternary symbol distribution for one neuron row."""
    max_abs = w_row.abs().max().item()
    threshold = TAU * max_abs

    n = w_row.numel()
    n_pos  = (w_row >  threshold).sum().item()
    n_neg  = (w_row < -threshold).sum().item()
    n_zero = n - n_pos - n_neg

    entropy = 0.0
    for count in [n_pos, n_neg, n_zero]:
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def select_entropy_guided(weight: torch.Tensor, fraction: float):
    """Select neurons in ascending order of ternary weight entropy."""
    n_neurons = weight.size(0)
    n_select  = math.floor(fraction * n_neurons)
    entropies = np.array([_ternary_entropy(weight[i]) for i in range(n_neurons)])
    indices = np.argsort(entropies)[:n_select].tolist()
    return sorted(indices)


# ──────────────────────────────────────────────────────────────
# High-level compression helpers
# ──────────────────────────────────────────────────────────────

def compress_fc1(model, fraction: float, strategy: str, seed: int = 42) -> nn.Module:
    """
    Partially ternarize FC1 neurons according to the chosen strategy.

    strategy: 'random' | 'l2_norm' | 'entropy_guided'
    Returns a deep-copied compressed model.
    """
    compressed = copy.deepcopy(model)
    weight = compressed.fc1.weight.data

    if strategy == "random":
        indices = select_random(weight, fraction, seed=seed)
    elif strategy == "l2_norm":
        indices = select_l2_norm(weight, fraction)
    elif strategy == "entropy_guided":
        indices = select_entropy_guided(weight, fraction)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    compressed.fc1.weight.data = ternarize_layer(weight, indices)
    return compressed


def compress_fc2_only(model) -> nn.Module:
    """Ternarize FC2 entirely, leave FC1 at FP32."""
    compressed = copy.deepcopy(model)
    weight = compressed.fc2.weight.data
    all_indices = list(range(weight.size(0)))
    compressed.fc2.weight.data = ternarize_layer(weight, all_indices)
    return compressed


# ──────────────────────────────────────────────────────────────
# Operation-count proxy
# ──────────────────────────────────────────────────────────────

def op_count(model, fraction: float, strategy: str, seed: int = 42) -> dict:
    """
    Compute operation-count proxy with zero-skipping.

    For FP32 baseline:
        OpCount = n_fc1_outputs * n_inputs  (FC1)  +  n_fc2_outputs * n_fc1_outputs  (FC2)

    For partial ternary FC1:
        Full neurons: unchanged
        Ternary neurons: count nnz weights (non-zero after ternarization)
        FC2 stays FP32

    Returns dict with keys: fp32_ops, method_ops, op_reduction_pct
    """
    # Dimensions
    in_dim  = model.fc1.weight.size(1)   # 784
    h_dim   = model.fc1.weight.size(0)   # 128
    out_dim = model.fc2.weight.size(0)   # 10

    fp32_ops = in_dim * h_dim + h_dim * out_dim  # baseline

    weight = model.fc1.weight.data

    if strategy == "random":
        ternary_indices = set(select_random(weight, fraction, seed=seed))
    elif strategy == "l2_norm":
        ternary_indices = set(select_l2_norm(weight, fraction))
    elif strategy == "entropy_guided":
        ternary_indices = set(select_entropy_guided(weight, fraction))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    method_ops = 0
    for i in range(h_dim):
        if i in ternary_indices:
            # Zero-skipping: count non-zero entries after ternarization
            w_tern = ternarize_weight_row(weight[i])
            method_ops += (w_tern != 0).sum().item()
        else:
            method_ops += in_dim  # full FP32 row

    method_ops += h_dim * out_dim  # FC2 unchanged

    op_reduction = 100.0 * (1.0 - method_ops / fp32_ops)
    return {
        "fp32_ops":      fp32_ops,
        "method_ops":    method_ops,
        "op_reduction":  op_reduction,
    }


def op_count_fc2_only(model) -> dict:
    """Operation count for FC2-only ternarization."""
    in_dim  = model.fc1.weight.size(1)
    h_dim   = model.fc1.weight.size(0)
    out_dim = model.fc2.weight.size(0)

    fp32_ops = in_dim * h_dim + h_dim * out_dim

    # FC1 unchanged; count nnz in ternary FC2
    weight_fc2 = model.fc2.weight.data
    fc2_ops = 0
    for i in range(out_dim):
        w_tern = ternarize_weight_row(weight_fc2[i])
        fc2_ops += (w_tern != 0).sum().item()

    method_ops = in_dim * h_dim + fc2_ops
    op_reduction = 100.0 * (1.0 - method_ops / fp32_ops)
    return {
        "fp32_ops":     fp32_ops,
        "method_ops":   method_ops,
        "op_reduction": op_reduction,
    }
