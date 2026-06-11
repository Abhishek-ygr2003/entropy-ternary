"""
Main pipeline: runs all compression phases and saves results.

Phases:
  Phase 0  - Train + save FP32 checkpoint
  Phase 1  - FP32 + INT8 baselines  (500-sample subset)
  Phase 2A - FC2-only ternarization  (500-sample subset)
  Phase 2B - L2-norm sweep 10%-60%   (500-sample subset)
  Phase 3  - Strategy comparison at 30% & 50%  (500-sample subset)
  Phase 4  - Full test set evaluation (10,000 samples)
  Phase 5  - Selection-seed stability (20 seeds, random 50%)
  Phase 6  - Training-seed robustness (3 training seeds)
"""

import json
import os
import copy
import numpy as np
import torch

from src.model      import MLP
from src.train      import set_seed, get_data_loaders, train, evaluate
from src.ternarize  import (
    simulate_int8, compress_fc1, compress_fc2_only,
    op_count, op_count_fc2_only,
    select_entropy_guided,
)

# ─── Config ───────────────────────────────────────────────────
TRAINING_SEED  = 42
EPOCHS         = 10
LR             = 1e-3
SUBSET_SIZE    = 500
DATA_DIR       = "./data"
CHECKPOINT_DIR = "./outputs/checkpoints"
RESULTS_DIR    = "./outputs/results"
FIGURES_DIR    = "./outputs/figures"


def ensure_dirs():
    for d in [CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


# ─── Phase 0: Train ───────────────────────────────────────────

def phase0_train(device):
    print("\n=== Phase 0: Training Base Model ===")
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"mlp_seed{TRAINING_SEED}.pt")

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists, loading from {ckpt_path}")
        model = MLP()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    set_seed(TRAINING_SEED)
    model = MLP()
    train_loader, _ = get_data_loaders(DATA_DIR, batch_size=256)
    model = train(model, train_loader, device, epochs=EPOCHS, lr=LR)
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")
    return model


# ─── Phase 1: Baselines ───────────────────────────────────────

def phase1_baselines(model, device):
    print("\n=== Phase 1: FP32 + INT8 Baselines (500-sample subset) ===")
    _, test_loader = get_data_loaders(DATA_DIR, subset_size=SUBSET_SIZE)

    fp32_acc = evaluate(model, test_loader, device)
    print(f"  FP32   : {fp32_acc:.2f}%")

    int8_model = simulate_int8(model)
    int8_acc   = evaluate(int8_model, test_loader, device)
    print(f"  INT8   : {int8_acc:.2f}%")

    results = {
        "FP32":  {"accuracy": fp32_acc, "drop": 0.0,                    "op_reduction": 0.0},
        "INT8":  {"accuracy": int8_acc, "drop": round(fp32_acc - int8_acc, 4), "op_reduction": 0.0},
    }
    save_json(results, os.path.join(RESULTS_DIR, "phase1_baselines.json"))
    return fp32_acc


# ─── Phase 2A: FC2-only ───────────────────────────────────────

def phase2a_fc2_only(model, fp32_acc, device):
    print("\n=== Phase 2A: FC2-Only Ternarization (500-sample subset) ===")
    _, test_loader = get_data_loaders(DATA_DIR, subset_size=SUBSET_SIZE)

    fc2_model = compress_fc2_only(model)
    acc = evaluate(fc2_model, test_loader, device)
    opred = op_count_fc2_only(model)["op_reduction"]
    drop = fp32_acc - acc
    print(f"  FC2-only ternary: Acc={acc:.2f}%  Drop={drop:.2f}%  OpRed={opred:.2f}%")

    results = {"accuracy": acc, "drop": drop, "op_reduction": opred}
    save_json(results, os.path.join(RESULTS_DIR, "phase2a_fc2_only.json"))


# ─── Phase 2B: L2 sweep ───────────────────────────────────────

def phase2b_l2_sweep(model, fp32_acc, device):
    print("\n=== Phase 2B: L2-Norm Sweep 10%-60% (500-sample subset) ===")
    _, test_loader = get_data_loaders(DATA_DIR, subset_size=SUBSET_SIZE)
    fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    results = {}

    for frac in fractions:
        comp = compress_fc1(model, fraction=frac, strategy="l2_norm")
        acc  = evaluate(comp, test_loader, device)
        opd  = op_count(model, fraction=frac, strategy="l2_norm")

        # Entropy of selected neurons (as reported in Table II)
        from src.ternarize import select_l2_norm, _ternary_entropy
        weight = model.fc1.weight.data
        indices = select_l2_norm(weight, frac)
        avg_entropy = float(np.mean([_ternary_entropy(weight[i]) for i in indices]))

        drop = fp32_acc - acc
        print(f"  {int(frac*100):3d}%  Acc={acc:.2f}%  Drop={drop:.2f}%  "
              f"OpRed={opd['op_reduction']:.2f}%  Entropy={avg_entropy:.4f}")

        results[f"{int(frac*100)}%"] = {
            "fraction": frac,
            "accuracy": acc,
            "drop": drop,
            "op_reduction": opd["op_reduction"],
            "avg_entropy": avg_entropy,
        }

    save_json(results, os.path.join(RESULTS_DIR, "phase2b_l2_sweep.json"))


# ─── Phase 3: Strategy comparison ────────────────────────────

def phase3_strategy_comparison(model, fp32_acc, device):
    print("\n=== Phase 3: Strategy Comparison at 30% & 50% (500-sample subset) ===")
    _, test_loader = get_data_loaders(DATA_DIR, subset_size=SUBSET_SIZE)
    strategies = ["random", "entropy_guided", "l2_norm"]
    fractions  = [0.30, 0.50]
    results    = {}

    for frac in fractions:
        results[f"{int(frac*100)}%"] = {}
        for strat in strategies:
            comp = compress_fc1(model, fraction=frac, strategy=strat, seed=TRAINING_SEED)
            acc  = evaluate(comp, test_loader, device)
            opd  = op_count(model, fraction=frac, strategy=strat, seed=TRAINING_SEED)
            drop = fp32_acc - acc
            print(f"  {int(frac*100)}%  {strat:18s}  Acc={acc:.2f}%  "
                  f"Drop={drop:.2f}%  OpRed={opd['op_reduction']:.2f}%")
            results[f"{int(frac*100)}%"][strat] = {
                "accuracy": acc, "drop": drop, "op_reduction": opd["op_reduction"]
            }

    save_json(results, os.path.join(RESULTS_DIR, "phase3_strategy_comparison.json"))


# ─── Phase 4: Full test set ───────────────────────────────────

def phase4_full_test(model, device):
    print("\n=== Phase 4: Full Test Set Evaluation (10,000 samples) ===")
    _, test_loader = get_data_loaders(DATA_DIR)  # no subset → 10,000
    strategies = ["random", "entropy_guided", "l2_norm"]
    frac = 0.50
    results = {}

    fp32_acc = evaluate(model, test_loader, device)
    print(f"  FP32             : {fp32_acc:.2f}%")
    results["FP32"] = {"accuracy": fp32_acc, "drop": 0.0, "op_reduction": 0.0}

    int8_acc = evaluate(simulate_int8(model), test_loader, device)
    drop_int8 = fp32_acc - int8_acc
    print(f"  INT8             : {int8_acc:.2f}%  Drop={drop_int8:.2f}%")
    results["INT8"] = {"accuracy": int8_acc, "drop": drop_int8, "op_reduction": 0.0}

    for strat in strategies:
        comp = compress_fc1(model, fraction=frac, strategy=strat, seed=TRAINING_SEED)
        acc  = evaluate(comp, test_loader, device)
        opd  = op_count(model, fraction=frac, strategy=strat, seed=TRAINING_SEED)
        drop = fp32_acc - acc
        label = f"{strat} 50%"
        print(f"  {label:22s}: Acc={acc:.2f}%  Drop={drop:.2f}%  OpRed={opd['op_reduction']:.2f}%")
        results[label] = {"accuracy": acc, "drop": drop, "op_reduction": opd["op_reduction"]}

    save_json(results, os.path.join(RESULTS_DIR, "phase4_full_test.json"))


# ─── Phase 5: Selection-seed stability ───────────────────────

def phase5_seed_stability(model, device, n_seeds: int = 20):
    print(f"\n=== Phase 5: Selection-Seed Stability (random 50%, {n_seeds} seeds) ===")
    _, test_loader = get_data_loaders(DATA_DIR)  # full 10,000-sample test set
    frac = 0.50
    accuracies = []

    for seed in range(n_seeds):
        comp = compress_fc1(model, fraction=frac, strategy="random", seed=seed)
        acc  = evaluate(comp, test_loader, device)
        accuracies.append(acc)
        print(f"  Seed {seed:2d}: {acc:.2f}%")

    mean_acc = float(np.mean(accuracies))
    std_acc  = float(np.std(accuracies))
    min_acc  = float(np.min(accuracies))
    max_acc  = float(np.max(accuracies))

    # Entropy-guided deterministic baseline
    comp_ent = compress_fc1(model, fraction=frac, strategy="entropy_guided")
    ent_acc  = evaluate(comp_ent, test_loader, device)

    print(f"\n  Random 50% -> Mean={mean_acc:.2f}%  std={std_acc:.2f}%  "
          f"Min={min_acc:.2f}%  Max={max_acc:.2f}%")
    print(f"  Entropy-guided (deterministic): {ent_acc:.2f}%")

    results = {
        "random_accuracies": accuracies,
        "mean":  mean_acc,
        "std":   std_acc,
        "min":   min_acc,
        "max":   max_acc,
        "entropy_guided_acc": ent_acc,
    }
    save_json(results, os.path.join(RESULTS_DIR, "phase5_seed_stability.json"))


# ─── Phase 6: Training-seed robustness ───────────────────────

def phase6_training_seed_robustness(device, training_seeds=(42, 7, 123)):
    print(f"\n=== Phase 6: Training-Seed Robustness ({training_seeds}) ===")
    _, test_loader = get_data_loaders(DATA_DIR)  # full 10,000-sample test set
    train_loader, _ = get_data_loaders(DATA_DIR, batch_size=256)
    frac = 0.50
    results = {}

    for tseed in training_seeds:
        set_seed(tseed)
        model = MLP()
        model = train(model, train_loader, device, epochs=EPOCHS, lr=LR)
        fp32_acc = evaluate(model, test_loader, device)

        row = {"fp32": fp32_acc}
        for strat in ["random", "entropy_guided", "l2_norm"]:
            comp = compress_fc1(model, fraction=frac, strategy=strat, seed=tseed)
            acc  = evaluate(comp, test_loader, device)
            row[strat] = acc
            print(f"  tseed={tseed}  {strat:18s}  FP32={fp32_acc:.2f}%  Compressed={acc:.2f}%")
        results[str(tseed)] = row

    save_json(results, os.path.join(RESULTS_DIR, "phase6_training_seed_robustness.json"))


# ─── Entry point ──────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ensure_dirs()

    model    = phase0_train(device)
    fp32_acc = phase1_baselines(model, device)
    phase2a_fc2_only(model, fp32_acc, device)
    phase2b_l2_sweep(model, fp32_acc, device)
    phase3_strategy_comparison(model, fp32_acc, device)
    phase4_full_test(model, device)
    phase5_seed_stability(model, device, n_seeds=20)
    phase6_training_seed_robustness(device)

    print("\nAll phases complete. Results in ./outputs/results/")


if __name__ == "__main__":
    main()
