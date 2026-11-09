"""
Improvement Experiments for the Entropy-Ternary Paper
=====================================================
1. Tau sensitivity sweep (tau = 0.1, 0.2, 0.3, 0.4, 0.5)
2. Actual inference latency measurement (FP32 vs ternary)
3. CIFAR-10 with a small CNN (harder benchmark)
4. INT8 baseline comparison at multiple bit-widths

All experiments use the existing trained MNIST checkpoint (seed 42).
Results saved to outputs/results/improvement_*.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import MLP
from src.train import set_seed, get_data_loaders, train, evaluate
from src.ternarize import (
    ternarize_weight_row,
    select_entropy_guided,
    select_random,
    compress_fc1,
    op_count,
    simulate_int8,
)

RESULTS_DIR = os.path.join("outputs", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


# =============================================================================
# EXPERIMENT 1: Tau Sensitivity Sweep
# =============================================================================
def experiment_tau_sweep():
    """Test different tau values (0.1 to 0.5) on the existing MNIST model."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TAU SENSITIVITY SWEEP")
    print("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load existing checkpoint
    ckpt_path = os.path.join("outputs", "checkpoints", "mlp_seed42.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found. Run pipeline.py first.")
        return None

    model = MLP().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Full test set
    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)

    # FP32 baseline
    fp32_acc = evaluate(model, test_loader, device)
    print(f"  FP32 Baseline Accuracy: {fp32_acc:.2f}%")

    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    fraction = 0.5  # 50% ternarization

    results = {
        "description": "Tau sensitivity sweep at 50% FC1 ternarization (full test set)",
        "fp32_accuracy": fp32_acc,
        "fraction": fraction,
        "tau_sweep": {}
    }

    for tau in tau_values:
        # We need to modify TAU globally since the module uses it
        import src.ternarize as tz
        original_tau = tz.TAU
        tz.TAU = tau

        try:
            model_entropy = compress_fc1(model, fraction, "entropy_guided")
            acc = evaluate(model_entropy, test_loader, device)
            ops = op_count(model, fraction, "entropy_guided")

            results["tau_sweep"][str(tau)] = {
                "accuracy": round(acc, 2),
                "drop_vs_fp32": round(fp32_acc - acc, 2),
                "op_reduction_pct": round(ops["op_reduction"], 2)
            }
            print(f"  tau={tau:.1f}: acc={acc:.2f}%, drop={fp32_acc-acc:.2f}pp, op_red={ops['op_reduction']:.2f}%")
        finally:
            tz.TAU = original_tau

    save_json(results, "improvement_tau_sweep.json")
    return results


# =============================================================================
# EXPERIMENT 2: Actual Inference Latency Measurement
# =============================================================================
def experiment_latency():
    """Measure actual wall-clock inference latency for FP32 vs ternary models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: INFERENCE LATENCY MEASUREMENT")
    print("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load existing checkpoint
    ckpt_path = os.path.join("outputs", "checkpoints", "mlp_seed42.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return None

    model = MLP().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Create ternary version
    fraction = 0.5
    model_entropy = compress_fc1(model, fraction, "entropy_guided")
    model_entropy.to(device)
    model_entropy.eval()

    # Generate a batch of test inputs (simulate realistic input)
    batch_size = 256
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)

    num_warmup = 100
    num_iters = 1000

    def measure_latency(model_to_time, n_warmup=num_warmup, n_iters=num_iters):
        """Measure average inference time in milliseconds per batch."""
        with torch.no_grad():
            # Warmup (important for GPU to stabilize clocks)
            for _ in range(n_warmup):
                _ = model_to_time(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed runs
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = model_to_time(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        avg_ms = (elapsed / n_iters) * 1000.0
        return avg_ms

    fp32_latency = measure_latency(model)
    ternary_latency = measure_latency(model_entropy)

    speedup = fp32_latency / ternary_latency if ternary_latency > 0 else float('inf')

    results = {
        "description": f"Wall-clock inference latency ({device.type.upper()}, batch_size=256)",
        "batch_size": batch_size,
        "num_iterations": num_iters,
        "fp32": {
            "avg_ms_per_batch": round(fp32_latency, 4),
            "avg_us_per_sample": round(fp32_latency * 1000 / batch_size, 2)
        },
        "ternary_50pct_entropy_guided": {
            "avg_ms_per_batch": round(ternary_latency, 4),
            "avg_us_per_sample": round(ternary_latency * 1000 / batch_size, 2)
        },
        "speedup_factor": round(speedup, 3),
        "note": "Note: PyTorch does not exploit zero-skipping natively in matmul kernels. "
                "The measured speedup reflects only reduced memory bandwidth effects. "
                "A custom ternary-aware kernel would yield larger gains."
    }

    print(f"  FP32:       {fp32_latency:.4f} ms/batch ({fp32_latency*1000/batch_size:.2f} us/sample)")
    print(f"  Ternary 50%: {ternary_latency:.4f} ms/batch ({ternary_latency*1000/batch_size:.2f} us/sample)")
    print(f"  Speedup:     {speedup:.3f}x")

    save_json(results, "improvement_latency.json")
    return results


# =============================================================================
# EXPERIMENT 3: CIFAR-10 with Small CNN (Harder Benchmark)
# =============================================================================
class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10: 3 conv layers + FC head."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def cifar_get_loaders(data_dir="./data_cifar", batch_size=128):
    """CIFAR-10 data loaders with standard augmentation for training."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def ternarize_cnn_fc(model, fraction, strategy="entropy_guided"):
    """Ternarize the FC layer of a CNN (the last linear layer)."""
    import src.ternarize as tz

    # Find the last Linear layer
    fc_layer = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            fc_layer = m  # keep going to get the LAST one

    if fc_layer is None:
        raise ValueError("No linear layer found")

    weight = fc_layer.weight.data.clone()
    n_neurons = weight.shape[0]
    indices = select_entropy_guided(weight, fraction) if strategy == "entropy_guided" else select_random(weight, fraction, seed=42)

    # Ternarize selected rows
    new_weight = weight.clone()
    for idx in indices:
        new_weight[idx] = ternarize_weight_row(weight[idx])

    model_copy = copy.deepcopy(model)
    fc_layer_ref = None
    for m in model_copy.modules():
        if isinstance(m, nn.Linear):
            fc_layer_ref = m

    fc_layer_ref.weight.data = new_weight
    return model_copy


def experiment_cifar10():
    """Train a small CNN on CIFAR-10 and apply entropy-guided ternarization."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CIFAR-10 SMALL CNN (HARDER BENCHMARK)")
    print("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Get data
    print("  Loading CIFAR-10...")
    train_loader, test_loader = cifar_get_loaders(data_dir="./data_cifar", batch_size=128)

    # Train model (30 epochs for decent accuracy on CPU)
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("  Training 30 epochs...")
    best_acc = 0.0
    for epoch in range(30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

        train_acc = 100.0 * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f}, train_acc={train_acc:.2f}%")

    # Evaluate FP32 baseline on full test set
    model.eval()
    fp32_acc = evaluate(model, test_loader, device)
    print(f"  CIFAR-10 FP32 Accuracy: {fp32_acc:.2f}%")

    # Apply ternarization to FC layer at different fractions
    results = {
        "description": "CIFAR-10 Small CNN (3 conv + FC) with entropy-guided FC ternarization",
        "architecture": "Conv(3->32)->Pool->Conv(32->64)->Pool->Conv(64->128)->FC(8192->256->10)",
        "fp32_accuracy": round(fp32_acc, 2),
        "ternarization_results": {}
    }

    for fraction in [0.25, 0.50, 0.75]:
        model_entropy = ternarize_cnn_fc(model, fraction, "entropy_guided")
        acc = evaluate(model_entropy, test_loader, device)
        results["ternarization_results"][str(int(fraction*100)) + "_pct"] = {
            "accuracy": round(acc, 2),
            "drop_vs_fp32": round(fp32_acc - acc, 2)
        }
        print(f"  FC ternarized {int(fraction*100)}%: acc={acc:.2f}%, drop={fp32_acc-acc:.2f}pp")

    # Also test random for comparison at 50%
    model_random = ternarize_cnn_fc(model, 0.50, "random")
    acc_rand = evaluate(model_random, test_loader, device)
    results["ternarization_results"]["random_50_pct"] = {
        "accuracy": round(acc_rand, 2),
        "drop_vs_fp32": round(fp32_acc - acc_rand, 2)
    }
    print(f"  FC ternarized 50% (random): acc={acc_rand:.2f}%, drop={fp32_acc-acc_rand:.2f}pp")

    save_json(results, "improvement_cifar10.json")
    return results


# =============================================================================
# EXPERIMENT 4: INT8 Baseline at Multiple Bit-Widths
# =============================================================================
def experiment_quantization_comparison():
    """Compare accuracy drops for INT8 vs ternary at equivalent compression levels."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: QUANTIZATION COMPARISON (INT8 vs Ternary)")
    print("=" * 70)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load existing checkpoint
    ckpt_path = os.path.join("outputs", "checkpoints", "mlp_seed42.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found.")
        return None

    model = MLP().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)

    fp32_acc = evaluate(model, test_loader, device)
    print(f"  FP32 Baseline: {fp32_acc:.2f}%")

    results = {
        "description": "INT8 vs Ternary comparison on MNIST MLP (full test set)",
        "fp32_accuracy": fp32_acc,
        "comparisons": {}
    }

    # INT8 simulation
    model_int8 = simulate_int8(model)
    int8_acc = evaluate(model_int8, test_loader, device)
    results["comparisons"]["INT8"] = {
        "accuracy": round(int8_acc, 2),
        "drop_vs_fp32": round(fp32_acc - int8_acc, 2),
        "memory_reduction_pct": 75.0,  # 4 bytes -> 1 byte per weight
        "op_reduction_pct": 0.0  # No structural reduction
    }
    print(f"  INT8: acc={int8_acc:.2f}%, drop={fp32_acc-int8_acc:.2f}pp, mem_red=75%, op_red=0%")

    # Ternary at various fractions
    import src.ternarize as tz
    for fraction in [0.25, 0.50, 0.75]:
        model_entropy = compress_fc1(model, fraction, "entropy_guided")
        acc = evaluate(model_entropy, test_loader, device)
        ops = op_count(model, fraction, "entropy_guided")

        # Memory reduction: ternary weights use ~1 byte vs 4 bytes for FP32
        # Only the ternarized portion gets memory savings
        mem_reduction = fraction * 75.0  # 75% saving on the ternarized portion
        results["comparisons"][f"Ternary_{int(fraction*100)}pct"] = {
            "accuracy": round(acc, 2),
            "drop_vs_fp32": round(fp32_acc - acc, 2),
            "op_reduction_pct": round(ops["op_reduction"], 2),
            "memory_savings_estimate_pct": round(mem_reduction, 1)
        }
        print(f"  Ternary {int(fraction*100)}%: acc={acc:.2f}%, drop={fp32_acc-acc:.2f}pp, op_red={ops['op_reduction']:.2f}%")

    save_json(results, "improvement_quant_comparison.json")
    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Starting Improvement Experiments...")
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    r1 = experiment_tau_sweep()
    r2 = experiment_latency()
    r3 = experiment_cifar10()
    r4 = experiment_quantization_comparison()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    if r1: print("  [1] Tau sweep: DONE")
    if r2: print("  [2] Latency: DONE")
    if r3: print("  [3] CIFAR-10 CNN: DONE")
    if r4: print("  [4] Quant comparison: DONE")
