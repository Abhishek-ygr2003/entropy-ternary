"""Run the remaining improvement experiments: tau sweep, latency, quant comparison.

CIFAR-10 results are already saved in outputs/results/improvement_cifar10.json,
so only experiments 1, 2 and 4 are run here (all use the MNIST checkpoint).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments_improvement import (
    experiment_tau_sweep,
    experiment_latency,
    experiment_quantization_comparison,
)


def main():
    r1 = experiment_tau_sweep()
    r2 = experiment_latency()
    r4 = experiment_quantization_comparison()

    print("\n" + "=" * 70)
    print("REMAINING EXPERIMENTS COMPLETE")
    print("=" * 70)
    if r1:
        print("  [1] Tau sweep: DONE")
    if r2:
        print("  [2] Latency: DONE")
    if r4:
        print("  [4] Quant comparison: DONE")


if __name__ == "__main__":
    main()
