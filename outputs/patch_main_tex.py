# -*- coding: utf-8 -*-
"""Integrate round-2 control results (R1-R4) into TMLR main.tex."""
import sys, io
sys.stdout.reconfigure(encoding="utf-8")

P = r"C:\Users\2026\Downloads\tmlr_submit\tmlr_submission\main.tex"
t = open(P, encoding="utf-8").read()

def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, f"anchor found {n}x (expected {count}): {old[:80]!r}"
    t = t.replace(old, new)

# ---------- 1. Abstract: tau sentence ----------
rep(
    r"We additionally find that $\tau=0.3$ remains the optimum when the sweep is run under corrected scaling and selected on a validation split carved from \emph{training} data.",
    r"Even the operating threshold does not survive: under corrected scaling the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$ (96.44\% vs.\ 95.68\%), and a validation split carved from \emph{training} data independently selects $\tau=0.2$---the original $0.3$ optimum was itself partly an artifact of the flawed scaling and test-set tuning.",
)

# ---------- 2. Intro: "optimality of tau survives" sentence ----------
rep(
    r"the optimality of the operating threshold $\tau$, in contrast, survives corrected scaling and training-holdout validation.",
    r"even the operating-threshold selection does not survive: under corrected scaling the sweep optimum moves to $\tau=0.2$, a choice that validation data held out from \emph{training} independently reproduces.",
)

# ---------- 3. Contribution 3 ----------
rep(
    r"\item A $\tau$-robustness result: the $\tau = 0.3$ optimum is re-confirmed under corrected scaling with validation data held out from \emph{training}, plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections~\ref{sec:cifar}--\ref{sec:latency}).",
    r"\item A negative $\tau$ result: under corrected scaling the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$, and a validation split held out from \emph{training} concurs---so the first-pass threshold selection was itself an artifact of the scaling convention and test-set tuning; plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections~\ref{sec:cifar}--\ref{sec:latency}).",
)

# ---------- 4. C2 additions: matched-op paragraph + multi-seed table ----------
c2_anchor = r"""\end{tabular}
\end{table}

\textbf{C3: The trivial baseline---matched magnitude pruning.}"""
c2_new = r"""\end{tabular}
\end{table}

\textbf{C2a: The reversal holds at matched operation count.}
Table~\ref{tab:twn} compares strategies at equal compression \emph{fraction}, but the strategies differ in how many operations each removes per fraction (entropy-50\% removes 42.33\% of FC1 operations vs.\ 38.44\% for random and 35.88\% for L2-norm, since entropy selects sparser rows). Entropy therefore ``pays'' for its slightly lower accuracy with more op reduction. We swept $f \in \{0.1,\ldots,0.9\}$ under corrected scaling for all three strategies and read off accuracy at the \emph{matched} 42.33\% op-reduction point (linear interpolation between neighboring sweep points): entropy-guided 95.68\%, L2-norm 96.50\%, random 96.88\%. The ordering reversal is thus not an artifact of unequal op counts---at matched operations the entropy criterion remains the worst of the three. The full accuracy-vs-op-reduction curves are released with the code.

\textbf{C2b: Training-seed variation under corrected scaling.}
The corrected-scaling comparison of Table~\ref{tab:twn} uses one training seed. Table~\ref{tab:seeds2} repeats it at $f=50\%$ on three independently trained models, with random selection reported as mean $\pm$ $\sigma$ over 20 selection seeds per training seed. Entropy-guided selection is clearly worst on seed 42 ($-1.25$ pp vs.\ the random mean) and statistically indistinguishable from random on seeds 7 and 123 (within one selection-seed $\sigma$); it never outperforms either baseline under corrected scaling.

\begin{table}[h]
\centering
\caption{Three training seeds under magnitude-restoring scaling, $f=50\%$, full test set. Random: mean $\pm$ $\sigma$ over 20 selection seeds}
\label{tab:seeds2}
\begin{tabular}{lcccc}
\toprule
Training seed & FP32 (\%) & Entropy (\%) & Random (\%) & L2-norm (\%) \\
\midrule
42  & 97.81 & 95.68 & $96.93 \pm 0.21$ & \textbf{96.81} \\
7   & 97.22 & 96.74 & $96.84 \pm 0.17$ & \textbf{96.75} \\
123 & 97.08 & 96.53 & $96.77 \pm 0.21$ & \textbf{96.61} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{C3: The trivial baseline---matched magnitude pruning.}"""
rep(c2_anchor, c2_new)

# ---------- 5. C4 rewrite ----------
old_c4 = r"""\textbf{C4: $\tau$ on a held-out validation split.}
To test whether $\tau = 0.3$ was a test-set artifact, we re-ran the $\tau$ sweep for entropy-guided selection with the last 2{,}000 test samples as validation and the first 8{,}000 as a disjoint test set. Validation accuracy peaks at $\tau = 0.3$ (95.20\%), independently selecting the same operating point, with test accuracy also peaking at 92.44\%. The unimodality of the sweep is therefore not a selection artifact---the one clearly robust finding from the original analysis, which we note does not sweeten the strategy comparison: a correctly tuned $\tau$ does not rescue the entropy criterion once C1--C3 are in place. We caution that the split-set accuracies are on 8{,}000 samples and are not directly comparable to the 10{,}000-sample tables elsewhere in the paper."""
new_c4 = r"""\textbf{C4: $\tau$ under corrected scaling, selected on training-holdout validation.}
The original validation split was carved from the \emph{test} set---flawed in principle---and the original sweep used the max-normalized scaling. We re-ran both. First, the $\tau$ sweep for entropy-guided selection at $f=50\%$ under magnitude-restoring scaling moves the optimum from $\tau=0.3$ (95.68\%) to $\tau=0.2$ (96.44\%); the first-pass unimodal curve at $0.3$ was partly shaped by the scaling artifact (the $\tau=0.1$ clip-floor amplification under max normalization contributed the low end of the original curve). Second, we retrained the seed-42 model on a 50k/10k split of the 60k \emph{training} set---the 10k validation split comes from training data and the 10k test set is untouched---and swept $\tau$ under corrected scaling. Validation accuracy selects $\tau=0.2$ (96.54\%), with test accuracy at that point 96.68\%, versus 96.25\% at $\tau=0.3$ (retrained FP32: 97.63\% test). The honest conclusion reverses our earlier claim: $\tau=0.3$ was not a robust operating point but a joint artifact of the scaling convention and test-set tuning. The protocol lesson stands, generalized: the threshold must be re-selected on training-holdout validation \emph{under the scaling convention actually deployed}, and it may move when either changes."""
rep(old_c4, new_c4)

# ---------- 6. tau sweep section addendum ----------
old_tau = r"We use $\tau = 0.3$ for all reported results, but we state two caveats explicitly. First, $\tau$ was selected on the full test set; a held-out validation split is required to claim a robust optimum rather than a fitted value. Second, the sweep covers five grid points at a single seed, and the 0.57-point gap between $\tau=0.2$ and $\tau=0.3$ is small relative to plausible run-to-run noise; we present $\tau = 0.3$ as the best point \emph{in our sweep}, not as a proven optimum."
new_tau = old_tau + r" Both caveats prove decisive: under the corrected magnitude-restoring scaling (Section~\ref{sec:controls}, C4) the optimum of this sweep moves to $\tau=0.2$ (96.44\% vs.\ 95.68\%), so the location---not only the selection protocol---of the sweep is scaling-dependent."
rep(old_tau, new_tau)

# ---------- 7. Conclusion rewrite ----------
old_concl = r"The operating threshold is the one finding that survives: a held-out validation split independently selects $\tau = 0.3$. Two further honest negatives bound the study---a directional failure on CIFAR-10 CNNs and a $1.008\times$ latency with stock PyTorch kernels that ignore structural zeros. We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but the demonstration that selection-criterion studies for partial quantization are unsafe without three controls: a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, and a matched-sparsity pruning baseline."
new_concl = r"Nothing about the criterion survives attribution---not even its operating point: under corrected scaling the $\tau$ optimum moves to $0.2$, and validation data held out from training concurs, so the original $\tau=0.3$ selection was itself an artifact. Two further honest negatives bound the study---a directional failure on CIFAR-10 CNNs and a $1.008\times$ latency with stock PyTorch kernels that ignore structural zeros. We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but the demonstration that selection-criterion studies for partial quantization are unsafe without four controls: a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, a matched-operation pruning baseline, and training-holdout threshold selection under the deployed scaling."
rep(old_concl, new_concl)

open(P, "w", encoding="utf-8", newline="").write(t)
print("All 7 integrations applied OK")
