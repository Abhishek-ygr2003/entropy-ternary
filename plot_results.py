"""
Generate all figures from the paper results.

Requires: matplotlib, numpy
Run after pipeline.py has completed.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = "./outputs/results"
FIGURES_DIR = "./outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "figure.figsize": (6, 4),
})


def load(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


# ── Figure 1: L2-Norm Sweep ───────────────────────────────────

def fig1_l2_sweep():
    data = load("phase2b_l2_sweep.json")
    fracs    = sorted(data.keys(), key=lambda x: int(x.strip("%")))
    accs     = [data[k]["accuracy"]     for k in fracs]
    opreds   = [data[k]["op_reduction"] for k in fracs]
    x_labels = [k for k in fracs]

    fig, ax1 = plt.subplots()
    x = np.arange(len(fracs))

    color_acc  = "#1f77b4"
    color_opred = "#d62728"

    ax1.plot(x, accs,   color=color_acc,   marker="s", label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)", color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Ternarization Fraction (%)")
    ax1.set_title("L2-Norm FC1 Sweep")

    ax2 = ax1.twinx()
    ax2.plot(x, opreds, color=color_opred, marker="^", linestyle="--", label="OpRed (%)")
    ax2.set_ylabel("Operation Reduction (%)", color=color_opred)
    ax2.tick_params(axis="y", labelcolor=color_opred)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_l2_sweep.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Figure 2: Strategy Comparison at 50% ─────────────────────

def fig2_strategy_comparison():
    data = load("phase4_full_test.json")
    strategies = ["random 50%", "entropy_guided 50%", "l2_norm 50%"]
    labels     = ["Random", "Entropy-guided", "L2-norm"]
    accs       = [data[s]["accuracy"]     for s in strategies]
    opreds     = [data[s]["op_reduction"] for s in strategies]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, accs,   width, label="Accuracy (%)",       color="#1f77b4")
    bars2 = ax.bar(x + width/2, opreds, width, label="Op Reduction (%)",   color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Strategy at 50% Ternarization (Full Test Set)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Strategy Comparison at 50% Ternarization")
    ax.legend(fontsize=9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_strategy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ── Figure 3: Seed Stability ──────────────────────────────────

def fig3_seed_stability():
    data = load("phase5_seed_stability.json")
    accs     = data["random_accuracies"]
    ent_acc  = data["entropy_guided_acc"]

    fig, ax = plt.subplots()
    ax.boxplot(accs, positions=[1], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor="#aec7e8"),
               medianprops=dict(color="navy", linewidth=2))
    ax.axhline(ent_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Entropy-guided (deterministic) {ent_acc:.2f}%")
    ax.set_xticks([1])
    ax.set_xticklabels(["Random (20 seeds)\nSelection Strategy"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Selection-Seed Variance: Random 50% Ternarization")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_seed_stability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved -> {path}")


def main():
    print("\n=== Generating Figures ===")
    try:
        fig1_l2_sweep()
    except FileNotFoundError:
        print("  [skip] phase2b_l2_sweep.json not found")

    try:
        fig2_strategy_comparison()
    except FileNotFoundError:
        print("  [skip] phase4_full_test.json not found")

    try:
        fig3_seed_stability()
    except FileNotFoundError:
        print("  [skip] phase5_seed_stability.json not found")

    print("Done.")


if __name__ == "__main__":
    main()
