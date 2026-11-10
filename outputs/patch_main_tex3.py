# -*- coding: utf-8 -*-
"""Third-round review fixes for TMLR main.tex."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

P = r"C:\Users\2026\Downloads\tmlr_submit\tmlr_submission\main.tex"
t = open(P, encoding="utf-8").read()

def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, f"anchor found {n}x (expected {count}): {old[:90]!r}"
    t = t.replace(old, new)

# ---- A. Abstract: reversal + pruning gap + tau ----
rep(
    r"""(2)~Once ternarization uses per-row magnitude-restoring scaling (as in Ternary Weight Networks), the strategy ordering reverses---L2-norm (96.81\%) and random (96.74\%) selection beat entropy-guided selection (95.68\%)---and the previously reported catastrophic L2-norm ``collapse'' proves to be an artifact of per-row max normalization. (3)~Unstructured magnitude pruning at matched operation count retains 97.80\% accuracy, about two points above the best corrected ternary configuration and four points above the originally reported one. Even the operating threshold does not survive: under corrected scaling the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$ (96.44\% vs.\ 95.68\%), and a validation split carved from \emph{training} data independently selects $\tau=0.2$---the original $0.3$ optimum was itself partly an artifact of the flawed scaling and test-set tuning.""",
    r"""(2)~Once ternarization uses per-row magnitude-restoring scaling (as in Ternary Weight Networks), the strategy ordering reverses at matched operation count---random (96.88\%) and L2-norm (96.50\%) selection beat entropy-guided selection (95.68\%), and random's advantage holds across three training seeds---and the previously reported catastrophic L2-norm ``collapse'' proves to be an artifact of per-row max normalization. (3)~Unstructured magnitude pruning at matched operation count retains 97.80\% accuracy: about one point above the best corrected ternary strategy, 4.8 points above the originally reported result, and a same-mask control attributes the deficit to \emph{which} weights are zeroed rather than to ternarizing survivors (which costs 0.06 pp, within noise). The operating threshold is scaling-dependent: under corrected scaling the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$ (96.44\% vs.\ 95.68\%), with validation data held out from \emph{training} concurring; a five-point, single-seed grid does not resolve the optimum further.""",
)

# ---- B. Intro premise: random variance ----
rep(
    r"""Random selection is simple but suffers from high run-to-run variance.""",
    r"""Random selection is simple; under the first-pass (max-normalized) scaling it showed high run-to-run variance ($\sigma \approx 3.9$ pp), though this variance shrinks to $\approx 0.2$ pp once magnitude-restoring scaling is adopted (Section~\ref{sec:controls}).""",
)

# ---- C. Intro tau-claim softening ----
rep(
    r"""even the operating-threshold selection does not survive: under corrected scaling the sweep optimum moves to $\tau=0.2$, a choice that validation data held out from \emph{training} independently reproduces.""",
    r"""and the operating-threshold selection does not survive unchanged: under corrected scaling the sweep optimum moves to $\tau=0.2$, a choice that validation data held out from \emph{training} independently reproduces---though a five-point, single-seed grid does not resolve the optimum between 0.2 and 0.3.""",
)

# ---- D. Contribution 3 softening ----
rep(
    r"""\item A negative $\tau$ result: under corrected scaling the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$, and a validation split held out from \emph{training} concurs---so the first-pass threshold selection was itself an artifact of the scaling convention and test-set tuning; plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections~\ref{sec:cifar}--\ref{sec:latency}).""",
    r"""\item A negative $\tau$ result: the sweep optimum moves from $\tau=0.3$ to $\tau=0.2$ under corrected scaling, with validation data held out from \emph{training} concurring---so the optimum is scaling-dependent and not resolved between 0.2 and 0.3 at this grid; plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections~\ref{sec:cifar}--\ref{sec:latency}).""",
)

# ---- E. Formal definition of corrected variant in Method ----
rep(
    r"""The selection-strategy comparisons in Section~\ref{sec:results} were run with this scaling, and the controls in Section~\ref{sec:controls} show that this choice materially affects which strategy appears best; a magnitude-restoring variant is evaluated explicitly there.""",
    r"""The selection-strategy comparisons in Section~\ref{sec:results} were run with this scaling, and the controls in Section~\ref{sec:controls} show that this choice materially affects which strategy appears best; a magnitude-restoring variant is evaluated explicitly there.

\textbf{Corrected (magnitude-restoring) variant.} The corrected variant keeps the thresholding rule identical---an entry survives if and only if $|w_i| > \tau W_{\max}$, so the zero-mask is the same in both variants---and changes only the survivor magnitudes: each surviving symbol $i \in S_j$ receives the row-level scale $\Delta_j = \frac{1}{|S_j|}\sum_{i \in S_j} |w_i|$ (the mean absolute weight of the row's surviving entries), so $w_e^{(i)} = w_t^{(i)} \cdot \Delta_j$. We use ``TWN-style'' advisedly: this borrows TWN's magnitude-restoring scale \citep{li2016ternary} but reuses our $\tau W_{\max}$ threshold rather than TWN's per-layer threshold optimization.""",
)

# ---- F. Related Work hedge ----
rep(
    r"""local entropy estimates serve as a practical proxy: low-entropy weight rows are ``easier'' to quantize because their information content is concentrated in fewer symbols.""",
    r"""local entropy estimates were hypothesized to serve as a practical proxy: low-entropy weight rows concentrate their information in fewer symbols and were expected to be easier to quantize. The controls in this paper revise that hypothesis: at our operating threshold, low entropy instead tracks sparsity (Section~\ref{sec:controls}).""",
)

# ---- G. 5.1 stale prose ----
rep(
    r"""it buys substantial compute reduction at a modest accuracy cost, whereas INT8 buys memory reduction only.""",
    r"""under the first-pass scaling it buys substantial compute reduction at what appears to be a modest accuracy cost; the controls (Section~\ref{sec:controls}) show the corrected accuracy cost is larger than the first-pass tables suggest, and matched magnitude pruning dominates the accuracy--op tradeoff.""",
)

# ---- H. 5.4 variance note ----
rep(
    r"""any fixed criterion, including norm-based ones, shares this determinism, and $\sigma = 0$ is therefore not itself a contribution.""",
    r"""any fixed criterion, including norm-based ones, shares this determinism, and $\sigma = 0$ is therefore not itself a contribution. We note that this $\sigma$ is measured under the original max-normalized scaling; under magnitude-restoring scaling the selection-seed variance of random selection shrinks to $\approx 0.2$ pp (Table~\ref{tab:seeds2}).""",
)

# ---- I. 5.7 INT8/ternary prose + Table 7 caption note ----
rep(
    r"""Partial ternarization trades roughly 4--5 points of accuracy for large, direct op-count reductions (22--61\%).""",
    r"""Under the first-pass scaling, partial ternarization trades roughly 3--5 points of accuracy for large, direct op-count reductions (22--61\%); under corrected scaling the cost at matched op counts is smaller (Table~\ref{tab:twn}) but still exceeds matched magnitude pruning (Section~\ref{sec:controls}).""",
)
rep(
    r"""ternarization is the right tool when compute (multiply-accumulate count) dominates, as in latency-constrained edge inference.""",
    r"""ternarization is the right tool when compute (multiply-accumulate count) dominates, as in latency-constrained edge inference---with the caveat that on this benchmark matched magnitude pruning dominates the ternary accuracy--op tradeoff (C3), so the compute advantage must be weighed against pruning's accuracy advantage.""",
)
rep(
    r"""\caption{INT8 vs.\ partial ternary quantization (MNIST MLP, full test set)}""",
    r"""\caption{INT8 vs.\ partial ternary quantization (MNIST MLP, full test set). Ternary rows use the original max-normalized scaling; corrected-scaling values are in Tables~\ref{tab:twn} and~\ref{tab:seeds2}}""",
)

# ---- J. C1 tautology acknowledgment ----
rep(
    r"""The \emph{apparent} entropy advantage over random selection (42.33\% vs.\ 38.44\% op reduction) is therefore a sparsity effect, not an information effect.""",
    r"""The \emph{apparent} entropy advantage over random selection (42.33\% vs.\ 38.44\% op reduction) is therefore a sparsity effect, not an information effect. We note that at a threshold whose surviving distribution is dominated by zeros, ternary entropy is analytically near-monotone in the non-zero fraction, so $\rho = +0.999$ is the \emph{expected} value rather than a discovery; this control is diagnostic confirmation that the criterion is masking sparsity, and C2--C3 carry the substantive attribution.""",
)

# ---- K. C2 paragraph + Table 10 caption: single-draw disclosure ----
rep(
    r"""At 50\%, random (96.74\%) and L2-norm (96.81\%) beat entropy-guided (95.68\%).""",
    r"""At 50\%, random (single selection seed: 96.74\%; 20-selection-seed mean $96.93 \pm 0.21$, Table~\ref{tab:seeds2}) and L2-norm (96.81\%) beat entropy-guided (95.68\%).""",
)
rep(
    r"""\caption{Strategy comparison with magnitude-restoring (TWN-style) scaling, full test set. Original (pre-correction) results in parentheses for reference}""",
    r"""\caption{Strategy comparison with magnitude-restoring (TWN-style) scaling, full test set. Original (pre-correction) results in parentheses for reference. The Random column reports a single selection seed; 20-seed statistics at 50\% are in Table~\ref{tab:seeds2}}""",
)

# ---- L/M. Table 11 caption reconciliation + C2b wording ----
rep(
    r"""\caption{Three training seeds under magnitude-restoring scaling, $f=50\%$, full test set. Random: mean $\pm$ $\sigma$ over 20 selection seeds}""",
    r"""\caption{Three training seeds under magnitude-restoring scaling, $f=50\%$, full test set. Random: mean $\pm$ $\sigma$ over 20 selection seeds. Seeds 7 and 123 were retrained for this control (the first-pass checkpoints were not retained), so their FP32 baselines differ from Table~\ref{tab:seeds}; all comparisons in this table use these retrained models and are internally consistent}""",
)
rep(
    r"""Entropy-guided selection is clearly worst on seed 42 ($-1.25$ pp vs.\ the random mean) and statistically indistinguishable from random on seeds 7 and 123 (within one selection-seed $\sigma$); it never outperforms either baseline under corrected scaling.""",
    r"""Entropy-guided selection is clearly worse on seed 42 ($-1.25$ pp vs.\ the random mean) and never better on seeds 7 and 123 (gaps of $0.10$ and $0.24$ pp, at most $1.2$ selection-seed $\sigma$). The honest summary is: never better than either baseline under corrected scaling, and clearly worse on one of three training seeds.""",
)

# ---- N. C3 gap numbers ----
rep(
    r"""Pruning dominates every ternary configuration on this task, though we emphasize the honest size of the gap: roughly two points against the corrected comparison, not five.""",
    r"""Pruning dominates every ternary configuration on this task. The honest size of the gap: about one point against the best corrected strategy at matched operations (random, 96.88\% interpolated from the released $f$-sweep; Table~\ref{tab:prune}), about two points against the entropy-corrected configuration (95.68\%), and 4.81 points against the originally reported one (92.99\%). At the 25\% op target, corrected L2-norm (97.48\%) comes within 0.31 points of pruning (97.79\%).""",
)

# ---- O. Table 12: corrected rows ----
rep(
    r"""\caption{Global magnitude pruning at matched op reduction vs.\ the matching ternary result (full test set, no retraining)}
\label{tab:prune}
\begin{tabular}{llcc}
\toprule
Op target & Method & Op Red.\ (\%) & Accuracy (\%) \\
\midrule
Entropy-25\% & Magnitude pruning & 22.2 & \textbf{97.79} \\
             & Entropy ternary    & 22.2 & 93.67 \\
Entropy-50\% & Magnitude pruning & 42.3 & \textbf{97.80} \\
             & Entropy ternary    & 42.3 & 92.99 \\
Entropy-75\% & Magnitude pruning & 61.3 & \textbf{97.45} \\
             & Entropy ternary    & 61.3 & 93.09 \\
Random-50\%  & Magnitude pruning & 38.4 & \textbf{97.76} \\
             & Random ternary     & 38.4 & 90.23 \\
\bottomrule
\end{tabular}""",
    r"""\caption{Global magnitude pruning at matched op reduction vs.\ ternary results under original and corrected scaling (full test set, no retraining; corrected rows interpolated from the released $f$-sweep at the matched op count)}
\label{tab:prune}
\begin{tabular}{llcc}
\toprule
Op target & Method & Op Red.\ (\%) & Accuracy (\%) \\
\midrule
Entropy-25\% & Magnitude pruning & 22.2 & \textbf{97.79} \\
             & Random ternary (corrected)   & 22.2 & 97.35 \\
             & L2-norm ternary (corrected)  & 22.2 & 97.33 \\
             & Entropy ternary (corrected)  & 22.2 & 96.43 \\
             & Entropy ternary (orig.\ scaling) & 22.2 & 93.67 \\
Entropy-50\% & Magnitude pruning & 42.3 & \textbf{97.80} \\
             & Random ternary (corrected)   & 42.3 & 96.88 \\
             & L2-norm ternary (corrected)  & 42.3 & 96.50 \\
             & Entropy ternary (corrected)  & 42.3 & 95.68 \\
             & Entropy ternary (orig.\ scaling) & 42.3 & 92.99 \\
Entropy-75\% & Magnitude pruning & 61.3 & \textbf{97.45} \\
             & Random ternary (corrected)   & 61.3 & 96.36 \\
             & L2-norm ternary (corrected)  & 61.3 & 95.98 \\
             & Entropy ternary (corrected)  & 61.3 & 95.80 \\
             & Entropy ternary (orig.\ scaling) & 61.3 & 93.09 \\
Random-50\%  & Magnitude pruning & 38.4 & \textbf{97.76} \\
             & Random ternary (corrected)   & 38.4 & 96.74 \\
             & Random ternary (orig.\ scaling)  & 38.4 & 90.23 \\
\bottomrule
\end{tabular}""",
)

# ---- P. C3a same-mask control ----
rep(
    r"""\textbf{C4: $\tau$ under corrected scaling, selected on training-holdout validation.}""",
    r"""\textbf{C3a: Same-mask control---what exactly does ternarization cost?}
Pruning differs from ternarization in two ways at once: \emph{which} weights are zeroed (the global smallest-magnitude mask vs.\ the selection criterion's row-structured mask) and what happens to survivors (kept FP32 vs.\ ternarized). To separate the two, we evaluated a variant that keeps the ternary model's \emph{exact} zero-mask---the threshold rule $|w_i| > \tau W_{\max}$ is scaling-independent, so the mask is identical in both scaling variants---while leaving all surviving weights in FP32. Table~\ref{tab:maskfp32} shows the decomposition at $f=50\%$. Ternarizing the survivors costs $0.06$ pp (within noise, and slightly \emph{favoring} ternarization: the per-row $\Delta_j$ slightly amplifies small survivors). The $2.18$ pp deficit relative to matched pruning therefore comes almost entirely from the \emph{mask choice}: the entropy criterion's row-structured zeroing discards more accuracy than the global smallest-magnitude mask at the same non-zero count. The max-normalized scaling artifact accounts for a further $2.69$ pp (92.99\% vs.\ 95.68\%). In short: at this benchmark, ternarizing survivors is nearly free; the accuracy deficit is bought by which weights are zeroed and, before correction, by the scaling.

\begin{table}[h]
\centering
\caption{Same-mask decomposition at $f=50\%$, entropy selection, full test set (seed 42)}
\label{tab:maskfp32}
\begin{tabular}{lcc}
\toprule
Variant & Accuracy (\%) & Gap to pruning (pp) \\
\midrule
FP32 baseline & 97.81 & $-0.01$ \\
Magnitude pruning (global mask) & \textbf{97.80} & --- \\
Same mask, FP32 survivors & 95.62 & $-2.18$ \\
Entropy ternary (corrected scaling) & 95.68 & $-2.12$ \\
Entropy ternary (original scaling) & 92.99 & $-4.81$ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{C4: $\tau$ under corrected scaling, selected on training-holdout validation.}""",
)

# ---- Q. C4 tau softening ----
rep(
    r"""The honest conclusion reverses our earlier claim: $\tau=0.3$ was not a robust operating point but a joint artifact of the scaling convention and test-set tuning. The protocol lesson stands, generalized: the threshold must be re-selected on training-holdout validation \emph{under the scaling convention actually deployed}, and it may move when either changes.""",
    r"""The honest conclusion softens our earlier claim: $\tau=0.3$ is not a robust operating point across scaling conventions---the optimum moves to $\tau=0.2$ under corrected scaling, and training-holdout validation concurs---but with one training seed and a five-point grid (a 0.43 pp test difference between 0.2 and 0.3) we cannot resolve the optimum further. The defensible claim is that the optimum is scaling-dependent and must be re-selected on training-holdout validation \emph{under the scaling convention actually deployed}.""",
)

# ---- R. Discussion mechanism hedge + pruning gap ----
rep(
    r"""the mechanism behind low entropy is mostly a single large outlier weight inflating $W_{\max}$, which pushes all other (smaller) weights below the threshold.""",
    r"""a pattern consistent with a single large outlier weight inflating $W_{\max}$, which pushes all other (smaller) weights below the threshold---though we did not test this mechanism directly and do not claim it as established.""",
)
rep(
    r"""At matched operation count, global magnitude pruning retains 97.80\% accuracy---about two points above the best scaling-corrected ternary configuration (95.68\%) and nearly five points above the originally reported one (92.99\%) (Tables~\ref{tab:prune} and~\ref{tab:twn}).""",
    r"""At matched operation count, global magnitude pruning retains 97.80\% accuracy---about one point above the best corrected strategy at matched operations (96.88\%, interpolated) and 4.81 points above the originally reported one (92.99\%) (Tables~\ref{tab:prune} and~\ref{tab:twn}).""",
)
rep(
    r"""but the compute benefit must be re-framed as an accuracy loss purchased at matched op counts, not a Pareto improvement.""",
    r"""but the compute benefit must be re-framed as an accuracy loss purchased at matched op counts, not a Pareto improvement. The same-mask control (C3a) sharpens this: the deficit is attributable to the zeroing mask, not to ternarizing survivors.""",
)

# ---- S. Conclusion ----
rep(
    r"""and matched-sparsity magnitude pruning retains 97.80\% accuracy (a 0.01-point drop), dominating every ternary strategy's accuracy--op tradeoff on this benchmark. Nothing about the criterion survives attribution---not even its operating point: under corrected scaling the $\tau$ optimum moves to $0.2$, and validation data held out from training concurs, so the original $\tau=0.3$ selection was itself an artifact.""",
    r"""and matched-sparsity magnitude pruning retains 97.80\% accuracy (a 0.01-point drop)---about one point above the best corrected ternary strategy at matched operations, with a same-mask control attributing the deficit to the zeroing mask rather than to ternarizing survivors ($0.06$ pp). The operating point is scaling-dependent: under corrected scaling the $\tau$ optimum moves to $0.2$, and validation data held out from training concurs, though a five-point, single-seed grid does not resolve 0.2 versus 0.3.""",
)
rep(
    r"""We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but the demonstration that selection-criterion studies for partial quantization are unsafe without four controls: a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, a matched-operation pruning baseline, and training-holdout threshold selection under the deployed scaling.""",
    r"""We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but what this case study suggests: selection-criterion studies for partial quantization are fragile without four controls---a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, a matched-operation pruning baseline (with a same-mask decomposition), and training-holdout threshold selection under the deployed scaling.""",
)

open(P, "w", encoding="utf-8", newline="").write(t)
print("All paper fixes applied OK")
