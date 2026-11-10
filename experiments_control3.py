"""
Control 3 (R5): Same-mask FP32-survivor control
================================================
Separates the two sources of damage in ternarization:
  (a) the zeroing MASK (which entries are zeroed), and
  (b) the SURVIVOR magnitudes (ternary delta_j scaling of survivors).

Variants at f=50%, entropy selection, seed-42 model, full test set:
  - ternary_original : max-normalized survivor scaling (first-pass)  -> 92.99
  - ternary_twn      : per-row delta_j scaling (corrected)           -> 95.68
  - mask_fp32        : SAME mask (zeroed entries identical -- the
                       threshold rule |w| > tau*Wmax is scaling-
                       independent), survivors kept at FP32
  - prune_matched    : global magnitude pruning at same op count     -> 97.80
                       (different mask: global smallest magnitudes)

Reading:
  ternary_twn vs mask_fp32   -> damage from ternarizing survivors
  mask_fp32 vs prune_matched -> damage from the mask choice
  ternary_original vs ternary_twn -> damage from max-normalized scaling
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.train import set_seed, get_data_loaders, train, evaluate
from src.model import MLP
from src.ternarize import select_entropy_guided
import src.ternarize as tz

RESULTS = os.path.join("outputs", "results")
os.makedirs(RESULTS, exist_ok=True)

def twn_row(w, tau=0.3):
    max_abs = w.abs().max().item()
    thr = tau * max_abs
    wt = torch.zeros_like(w)
    wt[w > thr] = 1.0
    wt[w < -thr] = -1.0
    nz = wt != 0
    delta = w.abs()[nz].mean().item() if nz.any() else 0.0
    return wt * delta

def orig_row(w, tau=0.3):
    """First-pass max-normalized row: wt * clip(|w|/Wmax, 0.1, 0.9)."""
    max_abs = w.abs().max().item()
    thr = tau * max_abs
    wt = torch.zeros_like(w)
    wt[w > thr] = 1.0
    wt[w < -thr] = -1.0
    p = (w.abs() / max_abs).clamp(0.1, 0.9)
    return wt * p

def make(model, fraction, row_fn, seed=42):
    c = copy.deepcopy(model)
    w = c.fc1.weight.data
    idx = select_entropy_guided(w, fraction)
    new_w = w.clone()
    for i in idx:
        new_w[i] = row_fn(w[i])
    c.fc1.weight.data = new_w
    return c

def mask_fp32(model, fraction, tau=0.3, seed=42):
    """Zero exactly the entries ternarization would zero (same mask),
    keep survivors at their original FP32 magnitudes."""
    c = copy.deepcopy(model)
    w = c.fc1.weight.data
    idx = select_entropy_guided(w, fraction)
    new_w = w.clone()
    for i in idx:
        row = w[i]
        max_abs = row.abs().max().item()
        thr = tau * max_abs
        keep = (row > thr) | (row < -thr)
        row = row.clone()
        row[~keep] = 0.0
        new_w[i] = row
    c.fc1.weight.data = new_w
    return c

def prune_global_matched(model, op_target_frac):
    """Global magnitude pruning of FC1 to match ternary op count.
    Ternary keeps (1-f) rows fully dense + f rows with nnz kept.
    We match the total non-zero count of the TWN ternary model."""
    c = copy.deepcopy(model)
    w = c.fc1.weight.data
    n_rows, d = w.shape
    # op count of ternary model (as in prior controls)
    idx = select_entropy_guided(w, 0.5)
    kept = sum(int((twn_row(w[i]) != 0).sum().item()) for i in idx)
    total_nnz_target = kept + (n_rows - len(idx)) * d
    # prune smallest |w| globally in FC1 to the same nnz
    flat = w.abs().flatten()
    k = int(flat.numel() - total_nnz_target)
    if k <= 0:
        return c
    thresh = torch.kthvalue(flat, k).values
    new_w = w.clone()
    new_w[w.abs() <= thresh] = 0.0
    c.fc1.weight.data = new_w
    return c

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(42)
    _, test_loader = get_data_loaders(data_dir="./data", batch_size=256)
    model = MLP().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/mlp_seed42.pt", map_location=device))
    model.eval()
    fp32 = evaluate(model, test_loader, device)
    print(f"FP32: {fp32:.2f}%")

    out = {"fp32": round(fp32, 2)}
    for name, m in [
        ("ternary_original", make(model, 0.5, orig_row)),
        ("ternary_twn", make(model, 0.5, twn_row)),
        ("mask_fp32", mask_fp32(model, 0.5)),
        ("prune_matched", prune_global_matched(model, 0.4233)),
    ]:
        acc = evaluate(m, test_loader, device)
        out[name] = round(acc, 2)
        print(f"{name:18s}: {acc:.2f}%")

    out["gaps"] = {
        "survivor_ternarization_damage": round(out["mask_fp32"] - out["ternary_twn"], 2),
        "mask_choice_damage": round(out["prune_matched"] - out["mask_fp32"], 2),
        "maxnorm_scaling_damage": round(out["ternary_twn"] - out["ternary_original"], 2),
    }
    with open(os.path.join(RESULTS, "control3_mask_fp32.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
