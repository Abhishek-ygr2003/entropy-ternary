# Attribution Controls for Neuron Selection in Partial Ternarization: A Negative Result

Reproducibility codebase for the paper:

> **Attribution Controls for Neuron Selection in Partial Ternarization: A Negative Result**  
> Abhishek Yadav GR  
> Garden City University, Bengaluru

---

> [!WARNING]
> **Dataset and Generalization Warning**
> This codebase and its findings have only been tested on the **MNIST** dataset using a simple two-layer MLP (784-128-10). The effectiveness of entropy-guided ternarization has not been validated on more complex datasets (e.g., CIFAR-10, ImageNet) or deeper architectures (e.g., ResNets, Transformers). Results may vary significantly when applied to other domains.

## Overview

This project implements and evaluates three neuron-selection strategies for post-training partial ternarization of a two-layer MLP on MNIST:

| Strategy | Deterministic | Notes |
|---|---|---|
| **Entropy-guided** | Yes (sigma = 0) | Selects neurons by ascending ternary entropy |
| **L2-norm** | Yes (sigma = 0) | Selects neurons by ascending weight norm |
| **Random** | No (sigma ~ 4%) | Uniform random subset |

**First-pass finding:** Entropy-guided selection achieved 42.33% operation-count reduction at 50% ternarization with 92.99% accuracy (vs FP32 97.81%), with zero selection-seed variance.

> [!IMPORTANT]
> **The first-pass advantage does not survive attribution controls** (see [`experiments_control.py`](experiments_control.py), round 1, [`experiments_control2.py`](experiments_control2.py), round 2, and [`experiments_control3.py`](experiments_control3.py), round 3; results in `outputs/results/control_*.json`, `control2_*.json`, and `control3_*.json`): ternary entropy correlates +0.999 with row sparsity (it is functionally a sparsity detector); under magnitude-restoring TWN scaling the strategy ordering reverses — at **matched** operation count random (96.88%) and L2-norm (96.50%) beat entropy (95.68%), and the reversal holds across 3 training seeds (entropy never wins); matched magnitude pruning dominates the accuracy–op tradeoff (97.80%, ≈1 point above the best corrected strategy at matched operations; a **same-mask control** attributes the deficit to *which* weights are zeroed, not to ternarizing survivors — same mask with FP32 survivors lands at 95.62%, within 0.06 pp of the ternary model); and the τ optimum moves from 0.3 to **0.2** under corrected scaling, with training-holdout validation concurring (a five-point, single-seed grid does not resolve 0.2 vs. 0.3 further).

---

## Architecture

The base model is a standard two-layer Multi-Layer Perceptron (MLP). Partial ternarization is applied selectively to the hidden neurons (FC1).

```mermaid
graph TD
    A[Input Image 28x28] -->|Flatten| B[784 Features]
    B --> C[FC1: 784 -> 128]
    C --> D[ReLU Activation]
    D --> E[FC2: 128 -> 10]
    E --> F[Output Logits]

    subgraph Partial Ternarization
        C -.-> G["Ternary Neurons (Selected via Strategy)"]
        C -.-> H["FP32 Neurons (Unchanged)"]
    end
```

---

## Theory

**What is Partial Ternarization?**
Ternarization usually quantizes neural network weights into three discrete values (e.g., `-1, 0, 1`). This reduces memory footprint and computational complexity. In this project, we explore **partial ternarization**, where only a fraction of the neurons in a layer are ternarized, while the rest remain in full precision (FP32). This creates a hybrid-precision layer that balances computational efficiency and model accuracy.

**Selection Strategies:**
The core question is *which* neurons to ternarize. We evaluate:
- **Random:** A naive baseline that picks neurons randomly.
- **L2-norm:** Picks neurons with the smallest weight magnitudes.
- **Entropy-guided:** Our novel approach. It calculates the Shannon entropy of the ternary symbol distribution for each neuron row and selects those with the lowest entropy (i.e., highly skewed distributions).

**Ternarization Formula:**
For a selected neuron weight row **W**:
```text
wt = +1   if w >  τ · max|W|
wt = −1   if w < −τ · max|W|
wt =  0   otherwise

τ = 0.3 (threshold)

weff = wt · pnz   where pnz = clip(|w| / max|W|, 0.1, 0.9)
weff = 0          when wt = 0
```
*Note: This specific hybrid formulation retains per-element continuous magnitude information (`pnz`), differentiating it from standard discrete TWN (Ternary Weight Networks).*

---

## Project Structure & Files

- `src/model.py`: Defines the `MLP` class (784-128-10 architecture).
- `src/train.py`: Contains standard PyTorch training and evaluation loops for MNIST.
- `src/ternarize.py`: The core logic. Contains functions for ternarizing weight rows, calculating operation-count proxies, simulating INT8, and implementing the three selection strategies (`select_random`, `select_l2_norm`, `select_entropy_guided`).
- `pipeline.py`: The main execution script. Orchestrates the training and the 6 experimental phases, saving outputs to JSON files.
- `experiments_control.py`: Round-1 attribution-control experiments (C1–C4): entropy–sparsity correlation, TWN-style magnitude-restoring scaling rerun, matched magnitude pruning, and τ selection on a test-carved validation split. Results land in `outputs/results/control_*.json`; a full console log is kept at `outputs/control_run.log`.
- `experiments_control2.py`: Round-2 attribution-control experiments under corrected scaling: τ sweep (optimum moves to 0.2), accuracy-vs-op-reduction curves with a matched-operation readout, τ selection on a validation split carved from **training** data (test set untouched), and a multi-training-seed comparison (seeds 42/7/123; random = mean ± σ over 20 selection seeds each). Results land in `outputs/results/control2_*.json`; full log at `outputs/control2_run.log`. Trained checkpoints for all three seeds are released under `outputs/checkpoints/`.
- `experiments_control3.py`: Round-3 same-mask decomposition (C3a): keeps the ternary model's exact zero-mask (the threshold rule |w| > τ·W_max is scaling-independent) while leaving survivors in FP32. Separates the two damage sources: ternarizing survivors costs 0.06 pp (noise), the mask choice costs 2.18 pp vs. matched pruning, and the max-norm scaling artifact accounts for a further 2.69 pp. Results in `outputs/results/control3_mask_fp32.json`; log at `outputs/control3_run.log`.
- `experiments_improvement.py`: τ sensitivity sweep, measured GPU latency, the CIFAR-10 CNN transfer probe, and the INT8 comparison.
- `plot_results.py`: Reads the generated JSON files and plots them using matplotlib.
- `tests/test_ternarize.py`: Comprehensive test suite verifying logic, matrix shapes, operations, and strict determinism of the entropy-guided strategy.

---

## Workflow

The experimental pipeline runs sequentially through several phases to evaluate the compression techniques.

```mermaid
flowchart TD
    Phase0[Phase 0: Train Base MLP 10 epochs] --> Phase1[Phase 1: Baselines FP32 & INT8]
    Phase1 --> Phase2A[Phase 2A: FC2-Only Ternarization]
    Phase1 --> Phase2B[Phase 2B: L2-norm Sweep 10% - 60%]
    Phase1 --> Phase3[Phase 3: Strategy Comparison at 30% & 50%]
    Phase1 --> Phase4[Phase 4: Full Test Set Evaluation]
    Phase1 --> Phase5[Phase 5: Selection-Seed Stability]
    Phase1 --> Phase6[Phase 6: Training-Seed Robustness]
    
    Phase0 -.-> Checkpoint[(Saved Model: mlp_seed42.pt)]
    Phase2B -.-> JSON[(Saved Results JSON)]
    Phase3 -.-> JSON
    Phase4 -.-> JSON
    Phase5 -.-> JSON
    Phase6 -.-> JSON
    JSON --> Plot[Generate Figures: plot_results.py]
```

---

## Setup & Usage

### Setup
```bash
git clone <your-repo-url>
cd entropy-ternary
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python pipeline.py
```
This will train the model, run all evaluations, and save the results in `outputs/results/`.

### Generate figures
```bash
python plot_results.py
```
Produces `fig1_l2_sweep.png`, `fig2_strategy_comparison.png`, and `fig3_seed_stability.png` in `outputs/figures/`.

### Run tests
```bash
pytest tests/ -v
```

---

## Results (Full Test Set, 10,000 samples)

Results are generated by running `python pipeline.py` and saved to `outputs/results/`.

Reference numbers from a reproducibility run (seed=42, 10 epochs, PyTorch 2.12, CPU):

| Method | Accuracy (%) | Drop (%) | OpRed (%) |
|---|---|---|---|
| FP32 | 97.75 | 0.00 | 0.00 |
| INT8 (simulated) | 97.73 | 0.02 | 0.00 |
| Random 50% | 89.86 | 7.89 | 38.49 |
| **Entropy-guided 50%** | 91.80 | 5.95 | 42.35 |
| L2-norm 50% | 60.44 | 37.31 | 35.90 |

Random 50% (20 seeds): mean = 88.81%, sigma = 4.13%, range [80.60%, 94.56%]

> **Note:** Exact numbers may vary slightly across PyTorch versions and hardware. Run the pipeline to generate your own results.

---

## Reproducibility Notes

- All experiments use `torch.manual_seed`, `numpy.random.seed`, and `random.seed` explicitly.
- The same trained checkpoint is reused for all strategy comparisons in a given run.
- Entropy-guided and L2-norm selections are **fully deterministic** under fixed weights — no seed needed.
- Random selection variance is evaluated across 20 independent seeds.
- Final reporting uses the **full 10,000-sample MNIST test set** as primary standard.

---

## Citation

```bibtex
@article{yadav2026entropy,
  title   = {Attribution Controls for Neuron Selection in Partial Ternarization: A Negative Result},
  author  = {Abhishek Yadav GR},
  year    = {2026},
  journal = {Preprint (code and results at github.com/Abhishek-ygr2003/entropy-ternary)}
}
```

---

## Acknowledgements
Bengaluru · PyTorch open-source community
