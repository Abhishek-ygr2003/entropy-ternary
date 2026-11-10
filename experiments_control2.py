"""
Control Experiments Round 2 — corrected-scaling attribution
===========================================================
All comparisons under TWN-style magnitude-restoring scaling
(weff = sign * delta_j, delta_j = mean |w| of surviving entries).

R1: tau sweep under TWN scaling (f=50%, entropy selection), full test set.
R2: Accuracy-vs-op-reduction curves for entropy/random/L2 under TWN scaling,
    f in {10..90%}; matched-op readout near the entropy-50% operating point.
R3: tau selection on a validation split carved from TRAINING data
    (retrain seed-42 model on 50k train / 10k val), then evaluate on the
    untouched 10k test set.
R4: Multi-training-seed comparison under TWN scaling at f=50%:
    entropy acc, random mean +/- sigma over 20 selection seeds, L2 acc.
    Seeds 7 and 123 are (re)trained here if checkpoints are missing.

Results -> outputs/results/control2_*.json
"""

import sys, os, json, math, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.model import MLP
from src.train import set_seed, get_data_loaders, train, evaluate
from src.ternarize import select_entropy_guided, select_random, select_l2_norm
import src.ternarize as tz

RESULTS_DIR = os.path.join("outputs", "results")
CKPT_DIR = os.path.join("outputs", "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)

TAUS = [0.1, 0.2, 0.3, 0.4, 0.5]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def twn_row(w, tau=0.3):
    """Magnitude-restoring ternarization: weff = sign * delta_j."""
    max_abs = w.abs().max().item()
    thr = tau * max_abs
    wt = torch.zeros_like(w)
    wt[w > thr] = 1.0
    wt[w < -thr] = -1.0
    nz = wt != 0
    delta = w.abs()[nz].mean().item() if nz.any() else 0.0
    return wt * delta


def compress_twn(model, fraction, strategy, seed=42, tau=0.3):
    """Ternarize FC1 rows (strategy-selected) with TWN scaling."""
    c = copy.deepcopy(model)
    w = c.fc1.weight.data
    if strategy == "entropy_guided":
        idx = select_entropy_guided(w, fraction)
    elif strategy == "random":
        idx = select_random(w, fraction, seed=seed)
    elif strategy == "l2_norm":
        idx = select_l2_norm(w, fraction)
    else:
        raise ValueError(strategy)
    new_w = w.clone()
    for i in idx:
        new_w[i] = twn_row(w[i], tau)
    c.fc1.weight.data = new_w
    return c


def op_reduction(model, fraction, strategy, seed=42, tau=0.3):
    w = model.fc1.weight.data
    n_rows, in_dim = w.size(0), w.size(1)
    if strategy == "entropy_guided":
        idx = set(select_entropy_guided(w, fraction))
    elif strategy == "random":
        idx = set(select_random(w, fraction, seed=seed))
    elif strategy == "l2_norm":
        idx = set(select_l2_norm(w, fraction))
    else:
        raise ValueError(strategy)
    method_ops = sum(
        int((twn_row(w[i], tau) != 0).sum().item()) if i in idx else in_dim
        for i in range(n_rows)) + n_rows * 10
    fp32_ops = in_dim * n_rows + n_rows * 10
    return 100.0 * (1.0 - method_ops / fp32_ops)


def ensure_checkpoint(seed, device):
    """Load training-seed checkpoint, retraining if absent."""
    path = os.path.join(CKPT_DIR, f"mlp_seed{seed}.pt")
    model = MLP().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"  [retrain] seed {seed} (checkpoint missing)")
        set_seed(seed)
        m = MLP().to(device)
        _, train_loader = get_data_loaders(data_dir="./data", batch_size=256)
        train(m, train_loader, device, epochs=10, lr=1e-3)
        torch.save(m.state_dict(), path)
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def r1_tau_twn(model, test_loader, device):
    print("\n" + "=" * 70)
    print("R1: TAU SWEEP UNDER TWN SCALING (entropy, f=50%, full test)")
    print("=" * 70)
    out = {}
    for tau in TAUS:
        tz.TAU = tau
        try:
            m = compress_twn(model, 0.5, "entropy_guided", tau=tau)
            acc = evaluate(m, test_loader, device)
        finally:
            tz.TAU = 0.3
        op = op_reduction(model, 0.5, "entropy_guided", tau=tau)
        out[str(tau)] = {"accuracy": round(acc, 2), "op_reduction": round(op, 2)}
        print(f"  tau={tau}: acc={acc:.2f}%, op_red={op:.2f}%")
    return out


def r2_curves(model, test_loader, device):
    print("\n" + "=" * 70)
    print("R2: ACCURACY-vs-OP-REDUCTION CURVES UNDER TWN SCALING")
    print("=" * 70)
    curves = {}
    for strat in ["entropy_guided", "random", "l2_norm"]:
        pts = []
        for f in FRACS:
            tz.TAU = 0.3
            m = compress_twn(model, f, strat)
            acc = evaluate(m, test_loader, device)
            op = op_reduction(model, f, strat)
            pts.append({"f": f, "op_reduction": round(op, 2), "accuracy": round(acc, 2)})
            print(f"  {strat:15s} f={f:.1f}: acc={acc:.2f}%, op_red={op:.2f}%")
        curves[strat] = pts

    # matched-op readout: accuracy of each strategy at op_red ~= entropy-50% point
    target = 42.33
    readout = {}
    for strat, pts in curves.items():
        best = min(pts, key=lambda p: abs(p["op_reduction"] - target))
        # linear interpolation between bracketing points
        spts = sorted(pts, key=lambda p: p["op_reduction"])
        acc_i = None
        for a, b in zip(spts, spts[1:]):
            if a["op_reduction"] <= target <= b["op_reduction"]:
                w = (target - a["op_reduction"]) / (b["op_reduction"] - a["op_reduction"])
                acc_i = a["accuracy"] + w * (b["accuracy"] - a["accuracy"])
                break
        readout[strat] = {
            "nearest_point": best,
            "interpolated_acc_at_42.33op": None if acc_i is None else round(acc_i, 2),
        }
        print(f"  matched-op {strat:15s}: nearest={best['op_reduction']}% op -> "
              f"acc={best['accuracy']}%; interp@42.33 = {readout[strat]['interpolated_acc_at_42.33op']}")
    return {"curves": curves, "matched_op_readout": readout}


def r3_trainholdout_val(device):
    print("\n" + "=" * 70)
    print("R3: TAU SELECTION WITH VALIDATION SPLIT FROM TRAINING DATA")
    print("=" * 70)
    set_seed(42)
    # rebuild datasets directly
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    n = len(train_ds)
    n_val = 10000
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=256, shuffle=True)
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=256, shuffle=False)
    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)

    model = MLP().to(device)
    train(model, train_loader, device, epochs=10, lr=1e-3)
    model.eval()
    fp32_test = evaluate(model, test_loader, device)
    fp32_val = evaluate(model, val_loader, device)
    print(f"  retrained FP32: val={fp32_val:.2f}%, test={fp32_test:.2f}%")

    sweep = {}
    for tau in TAUS:
        tz.TAU = tau
        try:
            m = compress_twn(model, 0.5, "entropy_guided", tau=tau)
            va, ta = evaluate(m, val_loader, device), evaluate(m, test_loader, device)
        finally:
            tz.TAU = 0.3
        sweep[str(tau)] = {"val_accuracy": round(va, 2), "test_accuracy": round(ta, 2)}
        print(f"  tau={tau}: val={va:.2f}%, test={ta:.2f}%")

    tau_star = max(sweep, key=lambda k: sweep[k]["val_accuracy"])
    return {
        "split": "10k validation held out from the 60k TRAINING set (disjoint from the 10k test set)",
        "fp32": {"val": round(fp32_val, 2), "test": round(fp32_test, 2)},
        "sweep": sweep,
        "tau_star_val": float(tau_star),
        "test_at_tau_star": sweep[tau_star]["test_accuracy"],
        "test_at_tau_03": sweep["0.3"]["test_accuracy"],
    }


def r4_multiseed(device, test_loader):
    print("\n" + "=" * 70)
    print("R4: MULTI-TRAINING-SEED COMPARISON UNDER TWN SCALING (f=50%)")
    print("=" * 70)
    out = {}
    for seed in [42, 7, 123]:
        model = ensure_checkpoint(seed, device)
        ent = evaluate(compress_twn(model, 0.5, "entropy_guided"), test_loader, device)
        l2 = evaluate(compress_twn(model, 0.5, "l2_norm"), test_loader, device)
        rnd = [evaluate(compress_twn(model, 0.5, "random", seed=s), test_loader, device)
               for s in range(20)]
        fp32 = evaluate(model, test_loader, device)
        out[str(seed)] = {
            "fp32": round(fp32, 2),
            "entropy": round(ent, 2),
            "l2_norm": round(l2, 2),
            "random_mean": round(float(np.mean(rnd)), 2),
            "random_sigma": round(float(np.std(rnd)), 2),
            "random_min": round(float(np.min(rnd)), 2),
            "random_max": round(float(np.max(rnd)), 2),
        }
        print(f"  seed {seed}: fp32={fp32:.2f} entropy={ent:.2f} "
              f"random={np.mean(rnd):.2f}+/-{np.std(rnd):.2f} l2={l2:.2f}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(42)
    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)
    model = ensure_checkpoint(42, device)
    fp32 = evaluate(model, test_loader, device)
    print(f"FP32 seed-42 baseline (full test): {fp32:.2f}%")

    r1 = r1_tau_twn(model, test_loader, device)
    with open(os.path.join(RESULTS_DIR, "control2_tau_twn.json"), "w") as f:
        json.dump(r1, f, indent=2)

    r2 = r2_curves(model, test_loader, device)
    with open(os.path.join(RESULTS_DIR, "control2_curves_twn.json"), "w") as f:
        json.dump(r2, f, indent=2)

    r3 = r3_trainholdout_val(device)
    with open(os.path.join(RESULTS_DIR, "control2_tau_trainholdout.json"), "w") as f:
        json.dump(r3, f, indent=2)

    r4 = r4_multiseed(device, test_loader)
    with open(os.path.join(RESULTS_DIR, "control2_multiseed_twn.json"), "w") as f:
        json.dump(r4, f, indent=2)

    print("\nROUND-2 CONTROLS COMPLETE")


if __name__ == "__main__":
    main()
