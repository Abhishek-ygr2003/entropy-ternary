"""
Control Experiments for the Entropy-Ternary Paper (reviewer response)
====================================================================
E1: Entropy-vs-sparsity correlation — does entropy just track row sparsity / magnitude?
E2: Per-row TWN-style magnitude-restoring scale factor — does the entropy advantage
    survive the corrected scaling?
E3: Global magnitude pruning at matched op reduction — the missing trivial baseline.
E4: tau selected on a held-out validation split, evaluated on a disjoint test split.

All experiments use the existing trained MNIST checkpoint (seed 42). No retraining.
Results saved to outputs/results/control_*.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.model import MLP
from src.train import set_seed, get_data_loaders, evaluate
from src.ternarize import (
    ternarize_weight_row,
    select_entropy_guided,
    select_random,
    select_l2_norm,
    op_count,
)

RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TAU = 0.3  # fixed operating point for the scaling-corrected experiments


def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


def load_model(device):
    ckpt_path = os.path.join("outputs", "checkpoints", "mlp_seed42.pt")
    model = MLP().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# Row statistics helpers
# ──────────────────────────────────────────────────────────────

def row_stats(w_row: torch.Tensor, tau: float = TAU):
    """Per-row diagnostics: entropy, nonzero fraction, extremes, norms."""
    max_abs = w_row.abs().max().item()
    threshold = tau * max_abs
    n = w_row.numel()
    n_pos = (w_row > threshold).sum().item()
    n_neg = (w_row < -threshold).sum().item()
    n_zero = n - n_pos - n_neg
    nnz = n_pos + n_neg

    entropy = 0.0
    for count in [n_pos, n_neg, n_zero]:
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return {
        "entropy": entropy,
        "nnz": nnz,
        "sparsity": 1.0 - nnz / n,
        "wmax": max_abs,
        "l2": w_row.norm(p=2).item(),
        "mean_abs": w_row.abs().mean().item(),
    }


def spearman(x, y):
    """Spearman rank correlation without scipy."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


# ──────────────────────────────────────────────────────────────
# E1: Entropy vs sparsity correlation diagnostics
# ──────────────────────────────────────────────────────────────

def experiment_correlation(model, test_loader, device):
    print("\n" + "=" * 70)
    print("E1: ENTROPY vs SPARSITY / MAGNITUDE CORRELATION")
    print("=" * 70)

    weight = model.fc1.weight.data
    n_rows = weight.size(0)

    ent, nnz, spar, wmax, l2, mabs = [], [], [], [], [], []
    for i in range(n_rows):
        s = row_stats(weight[i])
        ent.append(s["entropy"]); nnz.append(s["nnz"]); spar.append(s["sparsity"])
        wmax.append(s["wmax"]); l2.append(s["l2"]); mabs.append(s["mean_abs"])
    ent, nnz, spar, wmax, l2, mabs = (
        np.array(ent), np.array(nnz), np.array(spar), np.array(wmax), np.array(l2), np.array(mabs))

    corr = {
        "entropy_vs_nonzeros": spearman(ent, nnz),
        "entropy_vs_sparsity": spearman(ent, spar),
        "entropy_vs_wmax": spearman(ent, wmax),
        "entropy_vs_l2norm": spearman(ent, l2),
        "entropy_vs_meanabs": spearman(ent, mabs),
        "nonzeros_vs_wmax": spearman(nnz, wmax),
        "nnz_stats": {
            "mean_per_row": float(nnz.mean()), "min": int(nnz.min()), "max": int(nnz.max()),
        },
    }
    print(f"  Spearman correlations (tau=0.3):")
    for k, v in corr.items():
        if isinstance(v, float):
            print(f"    {k}: {v:+.3f}")

    # Does entropy selection ≈ picking the sparsest rows?
    frac = 0.5
    n_select = math.floor(frac * n_rows)
    ent_idx = set(select_entropy_guided(weight, frac))
    sparse_idx = set(np.argsort(nnz)[:n_select].tolist())
    overlap = len(ent_idx & sparse_idx) / n_select
    print(f"  Overlap(entropy-50%, sparsest-rows-50%): {overlap:.2%}")
    corr["overlap_entropy_vs_sparsest_50pct"] = round(overlap, 4)

    # Accuracy of "select sparsest rows" strategy (ternarize those rows, same rule)
    def compress_sparsest(model, frac):
        compressed = copy.deepcopy(model)
        weight = compressed.fc1.weight.data
        nnzs = np.array([row_stats(weight[i])["nnz"] for i in range(weight.size(0))])
        k = math.floor(frac * weight.size(0))
        idx = np.argsort(nnzs)[:k].tolist()
        new_w = weight.clone()
        for i in idx:
            new_w[i] = ternarize_weight_row(weight[i])
        compressed.fc1.weight.data = new_w
        return compressed

    res = {"correlations": corr, "sparsest_rows_strategy": {}}
    for frac in [0.25, 0.5, 0.75]:
        m = compress_sparsest(model, frac)
        acc = evaluate(m, test_loader, device)
        ops = op_count_sparsest(model, frac)
        res["sparsest_rows_strategy"][str(frac)] = {
            "accuracy": round(acc, 2),
            "op_reduction_pct": round(ops, 2),
        }
        print(f"  Sparsest-rows {frac:.0%}: acc={acc:.2f}%, op_red={ops:.2f}%")
    return res


def op_count_sparsest(model, frac):
    weight = model.fc1.weight.data
    n_rows = weight.size(0)
    nnzs = np.array([row_stats(weight[i])["nnz"] for i in range(n_rows)])
    k = math.floor(frac * n_rows)
    idx = set(np.argsort(nnzs)[:k].tolist())
    in_dim = weight.size(1)
    fp32_ops = in_dim * n_rows + n_rows * model.fc2.weight.size(0)
    method_ops = sum(nnzs[i] if i in idx else in_dim for i in range(n_rows)) + n_rows * 10
    return 100.0 * (1.0 - method_ops / fp32_ops)


# ──────────────────────────────────────────────────────────────
# E2: Per-row TWN-style magnitude-restoring scale factor
# ──────────────────────────────────────────────────────────────

def twn_ternarize_row(w_row: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """
    Magnitude-restoring ternarization (TWN-style):
        survivors = {i : |w_i| > tau * Wmax}
        delta_j   = mean(|w_i| for i in survivors)
        weff_i    = ± delta_j  for survivors, 0 otherwise
    This restores the row output scale in absolute units.
    """
    max_abs = w_row.abs().max().item()
    threshold = tau * max_abs
    wt = torch.zeros_like(w_row)
    wt[w_row > threshold] = 1.0
    wt[w_row < -threshold] = -1.0
    nonzero = wt != 0
    if nonzero.any():
        delta = w_row.abs()[nonzero].mean().item()
    else:
        delta = 0.0
    return wt * delta


def compress_twn(model, fraction, strategy, seed=42):
    """Ternarize FC1 rows selected by `strategy`, with TWN row scaling."""
    compressed = copy.deepcopy(model)
    weight = compressed.fc1.weight.data
    if strategy == "random":
        indices = select_random(weight, fraction, seed=seed)
    elif strategy == "l2_norm":
        indices = select_l2_norm(weight, fraction)
    elif strategy == "entropy_guided":
        indices = select_entropy_guided(weight, fraction)
    else:
        raise ValueError(strategy)
    new_w = weight.clone()
    for i in indices:
        new_w[i] = twn_ternarize_row(weight[i])
    compressed.fc1.weight.data = new_w
    return compressed


def experiment_twn_scaling(model, test_loader, device):
    print("\n" + "=" * 70)
    print("E2: MAGNITUDE-RESTORING (TWN-STYLE) SCALING RERUN")
    print("=" * 70)

    import src.ternarize as tz
    tz.TAU = TAU  # ensure selection threshold matches

    results = {"scaling": "delta_j = mean(|w| over survivors); weff = sign * delta_j", "fraction_results": {}}
    for frac in [0.25, 0.5, 0.75]:
        row = {}
        for strat in ["entropy_guided", "random", "l2_norm"]:
            m = compress_twn(model, frac, strat)
            acc = evaluate(m, test_loader, device)
            row[strat] = round(acc, 2)
            print(f"  TWN-scaled {strat:15s} @ {frac:.0%}: acc={acc:.2f}%")
        # op reduction unchanged in kind; compute for entropy for reference
        weight = model.fc1.weight.data
        idx = set(select_entropy_guided(weight, frac))
        in_dim = weight.size(1)
        n_rows = weight.size(0)
        ops = n_rows * 10 + sum(
            (twn_ternarize_row(weight[i]) != 0).sum().item() if i in idx else in_dim
            for i in range(n_rows))
        fp32_ops = in_dim * n_rows + n_rows * 10
        row["entropy_op_reduction_pct"] = round(100.0 * (1.0 - ops / fp32_ops), 2)
        results["fraction_results"][str(frac)] = row
    return results


# ──────────────────────────────────────────────────────────────
# E3: Global magnitude pruning at matched op reduction
# ──────────────────────────────────────────────────────────────

def experiment_magnitude_pruning(model, test_loader, device):
    print("\n" + "=" * 70)
    print("E3: GLOBAL MAGNITUDE PRUNING AT MATCHED OP REDUCTION")
    print("=" * 70)

    weight = model.fc1.weight.data
    in_dim, n_rows = weight.size(1), weight.size(0)
    fp32_ops = in_dim * n_rows + n_rows * 10

    # Target op counts from the ternary experiments
    targets = {}
    for frac, strat in [(0.25, "entropy"), (0.5, "entropy"), (0.75, "entropy"), (0.5, "random")]:
        import src.ternarize as tz
        tz.TAU = TAU
        if strat == "entropy":
            ops = op_count(model, frac, "entropy_guided")
        else:
            ops = op_count(model, frac, "random")
        targets[f"{strat}@{frac}"] = {
            "target_op_reduction_pct": round(ops["op_reduction"], 2),
            "target_fc1_nnz": int(ops["method_ops"] - n_rows * 10),
        }

    abs_w = weight.abs().flatten().cpu().numpy()

    def prune_to_nnz(model, target_nnz):
        pruned = copy.deepcopy(model)
        w = pruned.fc1.weight.data
        # threshold such that exactly target_nnz weights survive
        flat = w.abs().flatten()
        k_keep = target_nnz
        if k_keep >= flat.numel():
            return pruned
        thresh = torch.sort(flat)[0][flat.numel() - k_keep]
        mask = w.abs() > thresh
        w[~mask] = 0.0
        return pruned

    results = {"matched_pruning": {}}
    for key, t in targets.items():
        m = prune_to_nnz(model, t["target_fc1_nnz"])
        acc = evaluate(m, test_loader, device)
        # verify op count matches
        nnz_after = int((m.fc1.weight.data != 0).sum().item())
        ops_after = nnz_after + n_rows * 10
        op_red = 100.0 * (1.0 - ops_after / fp32_ops)
        strat, frac = key.split("@")
        ref = {"entropy@0.25": 93.67, "entropy@0.5": 92.99, "entropy@0.75": 93.09, "random@0.5": 90.23}[key]
        results["matched_pruning"][key] = {
            "target_op_reduction_pct": t["target_op_reduction_pct"],
            "achieved_op_reduction_pct": round(op_red, 2),
            "pruning_accuracy": round(acc, 2),
            "pruning_drop_vs_fp32": round(97.81 - acc, 2),
            "matching_ternary_accuracy": ref,
            "ternary_drop_vs_fp32": None if ref is None else round(97.81 - ref, 2),
        }
        print(f"  Pruned@{key:<15s}: acc={acc:.2f}% (ternary: {ref}%)")
    return results


# ──────────────────────────────────────────────────────────────
# E4: tau selected on a held-out validation split
# ──────────────────────────────────────────────────────────────

def experiment_tau_validation(model, device):
    print("\n" + "=" * 70)
    print("E4: TAU ON HELD-OUT VALIDATION SPLIT")
    print("=" * 70)

    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)
    dataset = test_loader.dataset
    n_total = len(dataset)  # 10000
    n_val = 2000
    val_idx = list(range(n_total - n_val, n_total))   # last 2000 = validation
    test_idx = list(range(n_total - n_val))           # first 8000 = test
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False)
    test_loader_split = DataLoader(Subset(dataset, test_idx), batch_size=256, shuffle=False)

    import src.ternarize as tz

    sweep = {}
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5]:
        original = tz.TAU
        tz.TAU = tau
        try:
            m = compress_fc1_tz(model, 0.5)
            val_acc = evaluate(m, val_loader, device)
            test_acc = evaluate(m, test_loader_split, device)
        finally:
            tz.TAU = original
        sweep[str(tau)] = {"val_accuracy": round(val_acc, 2), "test8000_accuracy": round(test_acc, 2)}
        print(f"  tau={tau}: val={val_acc:.2f}%, test8k={test_acc:.2f}%")

    tau_star = max(sweep, key=lambda k: sweep[k]["val_accuracy"])
    results = {
        "split": {"validation": "last 2000 of MNIST test set", "test": "first 8000 of MNIST test set"},
        "sweep": sweep,
        "tau_selected_on_validation": float(tau_star),
        "test_accuracy_at_selected_tau": sweep[tau_star]["test8000_accuracy"],
        "test_accuracy_at_tau03": sweep["0.3"]["test8000_accuracy"],
        "note": "Full-test-set numbers from other tables used all 10,000 samples with tau=0.3; "
                "these use a disjoint 8,000-sample test split and are not directly comparable.",
    }
    print(f"  tau* = {tau_star} (selected on validation)")
    return results


def compress_fc1_tz(model, fraction):
    from src.ternarize import compress_fc1
    return compress_fc1(model, fraction, "entropy_guided")


# ──────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)
    model = load_model(device)

    fp32_acc = evaluate(model, test_loader, device)
    print(f"FP32 baseline (full test): {fp32_acc:.2f}%")

    e1 = experiment_correlation(model, test_loader, device)
    save_json(e1, "control_entropy_sparsity.json")

    e2 = experiment_twn_scaling(model, test_loader, device)
    save_json(e2, "control_twn_scaling.json")
    # fix fp32 baseline key
    e2["fp32_accuracy_full_set"] = fp32_acc

    e3 = experiment_magnitude_pruning(model, test_loader, device)
    e3["fp32_accuracy_full_set"] = fp32_acc
    save_json(e3, "control_magnitude_pruning.json")

    e4 = experiment_tau_validation(model, device)
    save_json(e4, "control_tau_validation.json")

    print("\nALL CONTROL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
