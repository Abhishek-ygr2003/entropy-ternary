# -*- coding: utf-8 -*-
"""Third-round fixes for the public HTML mirror."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

P = r"C:\Users\2026\Downloads\entropy_ternary_paper.html"
t = open(P, encoding="utf-8").read()

def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, f"anchor found {n}x (expected {count}): {old[:90]!r}"
    t = t.replace(old, new)

# ---- TWN authors ----
rep("(as in TWN, Li &amp; Liu, 2016)", "(as in TWN, Li et al., 2016)")
rep("Li and Liu (2016) proposed Ternary Weight Networks (TWN)", "Li et al. (2016) proposed Ternary Weight Networks (TWN)")
rep("Unlike TWN (Li &amp; Liu, 2016)", "Unlike TWN (Li et al., 2016)")
rep("<p>F. Li and B. Liu. Ternary weight networks.", "<p>F. Li, B. Zhang, and B. Liu. Ternary weight networks.")

# ---- Abstract ----
rep(
    """(2)&nbsp;Once ternarization uses per-row magnitude-restoring scaling (as in Ternary Weight Networks), the strategy ordering reverses&mdash;L2-norm (96.81%) and random (96.74%) selection beat entropy-guided selection (95.68%)&mdash;and the previously reported catastrophic L2-norm &ldquo;collapse&rdquo; proves to be an artifact of per-row max normalization. (3)&nbsp;Unstructured magnitude pruning at matched operation count retains 97.80% accuracy, about two points above the best corrected ternary configuration and four points above the originally reported one. Even the operating threshold does not survive: under corrected scaling the sweep optimum moves from &tau;=0.3 to &tau;=0.2 (96.44% vs. 95.68%), and a validation split carved from <em>training</em> data independently selects &tau;=0.2&mdash;the original 0.3 optimum was itself partly an artifact of the flawed scaling and test-set tuning.""",
    """(2)&nbsp;Once ternarization uses per-row magnitude-restoring scaling (as in Ternary Weight Networks), the strategy ordering reverses at matched operation count&mdash;random (96.88%) and L2-norm (96.50%) selection beat entropy-guided selection (95.68%), and random's advantage holds across three training seeds&mdash;and the previously reported catastrophic L2-norm &ldquo;collapse&rdquo; proves to be an artifact of per-row max normalization. (3)&nbsp;Unstructured magnitude pruning at matched operation count retains 97.80% accuracy: about one point above the best corrected ternary strategy, 4.8 points above the originally reported result, and a same-mask control attributes the deficit to <em>which</em> weights are zeroed rather than to ternarizing survivors (which costs 0.06 pp, within noise). The operating threshold is scaling-dependent: under corrected scaling the sweep optimum moves from &tau;=0.3 to &tau;=0.2 (96.44% vs. 95.68%), with validation data held out from <em>training</em> concurring; a five-point, single-seed grid does not resolve the optimum further.""",
)

# ---- Intro premise ----
rep(
    "Random selection is simple but suffers from high run-to-run variance.",
    "Random selection is simple; under the first-pass (max-normalized) scaling it showed high run-to-run variance (&sigma; &asymp; 3.9 pp), though this variance shrinks to &asymp; 0.2 pp once magnitude-restoring scaling is adopted (Section 5.9).",
)

# ---- Intro para 2 ----
rep(
    "even the operating-threshold selection does not survive: under corrected scaling the sweep optimum moves to &tau;=0.2, a choice that validation data held out from <em>training</em> independently reproduces.",
    "and the operating-threshold selection does not survive unchanged: under corrected scaling the sweep optimum moves to &tau;=0.2, a choice that validation data held out from <em>training</em> independently reproduces&mdash;though a five-point, single-seed grid does not resolve the optimum between 0.2 and 0.3.",
)

# ---- Contribution 3 ----
rep(
    "<li>A negative &tau; result: under corrected scaling the sweep optimum moves from &tau;=0.3 to &tau;=0.2, and a validation split held out from <em>training</em> concurs&mdash;so the first-pass threshold selection was itself an artifact of the scaling convention and test-set tuning; plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections 5.6&ndash;5.8).</li>",
    "<li>A negative &tau; result: the sweep optimum moves from &tau;=0.3 to &tau;=0.2 under corrected scaling, with validation data held out from <em>training</em> concurring&mdash;so the optimum is scaling-dependent and not resolved between 0.2 and 0.3 at this grid; plus a negative CIFAR-10 CNN transfer, an INT8 comparison, and latency measurements showing no kernel-exploitable gain (Sections 5.6&ndash;5.8).</li>",
)

# ---- Method: corrected-variant definition ----
rep(
    """a magnitude-restoring variant is evaluated explicitly there.</p>""",
    """a magnitude-restoring variant is evaluated explicitly there.</p>

<p><strong>Corrected (magnitude-restoring) variant.</strong> The corrected variant keeps the thresholding rule identical&mdash;an entry survives if and only if |w<sub>i</sub>| &gt; &tau;W<sub>max</sub>, so the zero-mask is the same in both variants&mdash;and changes only the survivor magnitudes: each surviving symbol i &isin; S<sub>j</sub> receives the row-level scale &Delta;<sub>j</sub> = (1/|S<sub>j</sub>|) &Sigma;<sub>i&isin;S<sub>j</sub></sub> |w<sub>i</sub>| (the mean absolute weight of the row's surviving entries), so w<sub>e</sub><sup>(i)</sup> = w<sub>t</sub><sup>(i)</sup> &middot; &Delta;<sub>j</sub>. We use &ldquo;TWN-style&rdquo; advisedly: this borrows TWN's magnitude-restoring scale (Li et al., 2016) but reuses our &tau;W<sub>max</sub> threshold rather than TWN's per-layer threshold optimization.</p>""",
)

# ---- Related Work hedge ----
rep(
    """local entropy estimates serve as a practical proxy: low-entropy weight rows are &ldquo;easier&rdquo; to quantize because their information content is concentrated in fewer symbols.""",
    """local entropy estimates were hypothesized to serve as a practical proxy: low-entropy weight rows concentrate their information in fewer symbols and were expected to be easier to quantize. The controls in this paper revise that hypothesis: at our operating threshold, low entropy instead tracks sparsity (Section 5.9).""",
)

# ---- 5.1 prose ----
rep(
    """it buys substantial compute reduction at a modest accuracy cost, whereas INT8 buys memory reduction only.""",
    """under the first-pass scaling it buys substantial compute reduction at what appears to be a modest accuracy cost; the controls (Section 5.9) show the corrected accuracy cost is larger than the first-pass tables suggest, and matched magnitude pruning dominates the accuracy&ndash;op tradeoff.""",
)

# ---- 5.4 variance note ----
rep(
    """any fixed criterion, including norm-based ones, shares this determinism, and &sigma; = 0 is therefore not itself a contribution.""",
    """any fixed criterion, including norm-based ones, shares this determinism, and &sigma; = 0 is therefore not itself a contribution. We note that this &sigma; is measured under the original max-normalized scaling; under magnitude-restoring scaling the selection-seed variance of random selection shrinks to &asymp; 0.2 pp (Table 11).""",
)

# ---- 5.7 prose + Table 7 caption ----
rep(
    """Partial ternarization trades roughly 4&ndash;5 points of accuracy for large, direct op-count reductions (22&ndash;61%).""",
    """Under the first-pass scaling, partial ternarization trades roughly 3&ndash;5 points of accuracy for large, direct op-count reductions (22&ndash;61%); under corrected scaling the cost at matched op counts is smaller (Table 10) but still exceeds matched magnitude pruning (Section 5.9).""",
)
rep(
    """ternarization is the right tool when compute (multiply-accumulate count) dominates, as in latency-constrained edge inference.""",
    """ternarization is the right tool when compute (multiply-accumulate count) dominates, as in latency-constrained edge inference&mdash;with the caveat that on this benchmark matched magnitude pruning dominates the ternary accuracy&ndash;op tradeoff (C3), so the compute advantage must be weighed against pruning's accuracy advantage.""",
)
rep(
    """Table 7: INT8 vs. partial ternary quantization (MNIST MLP, full test set)""",
    """Table 7: INT8 vs. partial ternary quantization (MNIST MLP, full test set). Ternary rows use the original max-normalized scaling; corrected-scaling values are in Tables 10 and 11""",
)

# ---- C1 tautology ----
rep(
    """The <em>apparent</em> entropy advantage over random selection (42.33% vs. 38.44% op reduction) is therefore a sparsity effect, not an information effect.""",
    """The <em>apparent</em> entropy advantage over random selection (42.33% vs. 38.44% op reduction) is therefore a sparsity effect, not an information effect. We note that at a threshold whose surviving distribution is dominated by zeros, ternary entropy is analytically near-monotone in the non-zero fraction, so &rho; = +0.999 is the <em>expected</em> value rather than a discovery; this control is diagnostic confirmation that the criterion is masking sparsity, and C2&ndash;C3 carry the substantive attribution.""",
)

# ---- C2 text + Table 10 caption ----
rep(
    """At 50%, random (96.74%) and L2-norm (96.81%) beat entropy-guided (95.68%).""",
    """At 50%, random (single selection seed: 96.74%; 20-selection-seed mean 96.93 &plusmn; 0.21, Table 11) and L2-norm (96.81%) beat entropy-guided (95.68%).""",
)
rep(
    """Table 10: Strategy comparison with magnitude-restoring (TWN-style) scaling, full test set. Original (pre-correction) results in parentheses for reference""",
    """Table 10: Strategy comparison with magnitude-restoring (TWN-style) scaling, full test set. Original (pre-correction) results in parentheses for reference. The Random column reports a single selection seed; 20-seed statistics at 50% are in Table 11""",
)

# ---- Table 11 caption + C2b ----
rep(
    """Table 11: Three training seeds under magnitude-restoring scaling, f=50%, full test set. Random: mean &plusmn; &sigma; over 20 selection seeds""",
    """Table 11: Three training seeds under magnitude-restoring scaling, f=50%, full test set. Random: mean &plusmn; &sigma; over 20 selection seeds. Seeds 7 and 123 were retrained for this control (the first-pass checkpoints were not retained), so their FP32 baselines differ from Table 5; all comparisons in this table use these retrained models and are internally consistent""",
)
rep(
    """Entropy-guided selection is clearly worst on seed 42 (&minus;1.25 pp vs. the random mean) and statistically indistinguishable from random on seeds 7 and 123 (within one selection-seed &sigma;); it never outperforms either baseline under corrected scaling.""",
    """Entropy-guided selection is clearly worse on seed 42 (&minus;1.25 pp vs. the random mean) and never better on seeds 7 and 123 (gaps of 0.10 and 0.24 pp, at most 1.2 selection-seed &sigma;). The honest summary is: never better than either baseline under corrected scaling, and clearly worse on one of three training seeds.""",
)

# ---- C3 numbers ----
rep(
    """Pruning dominates every ternary configuration on this task, though we emphasize the honest size of the gap: roughly two points against the corrected comparison, not five.""",
    """Pruning dominates every ternary configuration on this task. The honest size of the gap: about one point against the best corrected strategy at matched operations (random, 96.88% interpolated from the released f-sweep; Table 12), about two points against the entropy-corrected configuration (95.68%), and 4.81 points against the originally reported one (92.99%). At the 25% op target, corrected L2-norm (97.48%) comes within 0.31 points of pruning (97.79%).""",
)

# ---- Table 12 rows ----
rep(
    """<tr><td>Entropy-25%</td><td>Magnitude pruning</td><td>22.2</td><td><strong>97.79</strong></td></tr>
<tr><td></td><td>Entropy ternary</td><td>22.2</td><td>93.67</td></tr>
<tr><td>Entropy-50%</td><td>Magnitude pruning</td><td>42.3</td><td><strong>97.80</strong></td></tr>
<tr><td></td><td>Entropy ternary</td><td>42.3</td><td>92.99</td></tr>
<tr><td>Entropy-75%</td><td>Magnitude pruning</td><td>61.3</td><td><strong>97.45</strong></td></tr>
<tr><td></td><td>Entropy ternary</td><td>61.3</td><td>93.09</td></tr>
<tr><td>Random-50%</td><td>Magnitude pruning</td><td>38.4</td><td><strong>97.76</strong></td></tr>
<tr><td></td><td>Random ternary</td><td>38.4</td><td>90.23</td></tr>""",
    """<tr><td>Entropy-25%</td><td>Magnitude pruning</td><td>22.2</td><td><strong>97.79</strong></td></tr>
<tr><td></td><td>Random ternary (corrected)</td><td>22.2</td><td>97.35</td></tr>
<tr><td></td><td>L2-norm ternary (corrected)</td><td>22.2</td><td>97.33</td></tr>
<tr><td></td><td>Entropy ternary (corrected)</td><td>22.2</td><td>96.43</td></tr>
<tr><td></td><td>Entropy ternary (orig. scaling)</td><td>22.2</td><td>93.67</td></tr>
<tr><td>Entropy-50%</td><td>Magnitude pruning</td><td>42.3</td><td><strong>97.80</strong></td></tr>
<tr><td></td><td>Random ternary (corrected)</td><td>42.3</td><td>96.88</td></tr>
<tr><td></td><td>L2-norm ternary (corrected)</td><td>42.3</td><td>96.50</td></tr>
<tr><td></td><td>Entropy ternary (corrected)</td><td>42.3</td><td>95.68</td></tr>
<tr><td></td><td>Entropy ternary (orig. scaling)</td><td>42.3</td><td>92.99</td></tr>
<tr><td>Entropy-75%</td><td>Magnitude pruning</td><td>61.3</td><td><strong>97.45</strong></td></tr>
<tr><td></td><td>Random ternary (corrected)</td><td>61.3</td><td>96.36</td></tr>
<tr><td></td><td>L2-norm ternary (corrected)</td><td>61.3</td><td>95.98</td></tr>
<tr><td></td><td>Entropy ternary (corrected)</td><td>61.3</td><td>95.80</td></tr>
<tr><td></td><td>Entropy ternary (orig. scaling)</td><td>61.3</td><td>93.09</td></tr>
<tr><td>Random-50%</td><td>Magnitude pruning</td><td>38.4</td><td><strong>97.76</strong></td></tr>
<tr><td></td><td>Random ternary (corrected)</td><td>38.4</td><td>96.74</td></tr>
<tr><td></td><td>Random ternary (orig. scaling)</td><td>38.4</td><td>90.23</td></tr>""",
)
rep(
    """Table 12: Global magnitude pruning at matched op reduction vs. the matching ternary result (full test set, no retraining)""",
    """Table 12: Global magnitude pruning at matched op reduction vs. ternary results under original and corrected scaling (full test set, no retraining; corrected rows interpolated from the released f-sweep at the matched op count)""",
)

# ---- C3a ----
rep(
    """<p><strong>C4: &tau; under corrected scaling, selected on training-holdout validation.</strong>""",
    """<p><strong>C3a: Same-mask control&mdash;what exactly does ternarization cost?</strong> Pruning differs from ternarization in two ways at once: <em>which</em> weights are zeroed (the global smallest-magnitude mask vs. the selection criterion's row-structured mask) and what happens to survivors (kept FP32 vs. ternarized). To separate the two, we evaluated a variant that keeps the ternary model's <em>exact</em> zero-mask&mdash;the threshold rule |w<sub>i</sub>| &gt; &tau;W<sub>max</sub> is scaling-independent, so the mask is identical in both scaling variants&mdash;while leaving all surviving weights in FP32. Table 13 shows the decomposition at f=50%. Ternarizing the survivors costs 0.06 pp (within noise, and slightly <em>favoring</em> ternarization: the per-row &Delta;<sub>j</sub> slightly amplifies small survivors). The 2.18 pp deficit relative to matched pruning therefore comes almost entirely from the <em>mask choice</em>: the entropy criterion's row-structured zeroing discards more accuracy than the global smallest-magnitude mask at the same non-zero count. The max-normalized scaling artifact accounts for a further 2.69 pp (92.99% vs. 95.68%). In short: at this benchmark, ternarizing survivors is nearly free; the accuracy deficit is bought by which weights are zeroed and, before correction, by the scaling.</p>

<div class="table-caption">Table 13: Same-mask decomposition at f=50%, entropy selection, full test set (seed 42)</div>
<table>
<tr><th>Variant</th><th>Accuracy (%)</th><th>Gap to pruning (pp)</th></tr>
<tr><td>FP32 baseline</td><td>97.81</td><td>&minus;0.01</td></tr>
<tr><td>Magnitude pruning (global mask)</td><td><strong>97.80</strong></td><td>&mdash;</td></tr>
<tr><td>Same mask, FP32 survivors</td><td>95.62</td><td>&minus;2.18</td></tr>
<tr><td>Entropy ternary (corrected scaling)</td><td>95.68</td><td>&minus;2.12</td></tr>
<tr><td>Entropy ternary (original scaling)</td><td>92.99</td><td>&minus;4.81</td></tr>
</table>

<p><strong>C4: &tau; under corrected scaling, selected on training-holdout validation.</strong>""",
)

# ---- C4 softening ----
rep(
    """The honest conclusion reverses our earlier claim: &tau;=0.3 was not a robust operating point but a joint artifact of the scaling convention and test-set tuning. The protocol lesson stands, generalized: the threshold must be re-selected on training-holdout validation <em>under the scaling convention actually deployed</em>, and it may move when either changes.""",
    """The honest conclusion softens our earlier claim: &tau;=0.3 is not a robust operating point across scaling conventions&mdash;the optimum moves to &tau;=0.2 under corrected scaling, and training-holdout validation concurs&mdash;but with one training seed and a five-point grid (a 0.43 pp test difference between 0.2 and 0.3) we cannot resolve the optimum further. The defensible claim is that the optimum is scaling-dependent and must be re-selected on training-holdout validation <em>under the scaling convention actually deployed</em>.""",
)

# ---- Discussion ----
rep(
    """the mechanism behind low entropy is mostly a single large outlier weight inflating W<sub>max</sub>, which pushes all other (smaller) weights below the threshold.""",
    """a pattern consistent with a single large outlier weight inflating W<sub>max</sub>, which pushes all other (smaller) weights below the threshold&mdash;though we did not test this mechanism directly and do not claim it as established.""",
)
rep(
    """At matched operation count, global magnitude pruning retains 97.80% accuracy&mdash;about two points above the best scaling-corrected ternary configuration (95.68%) and nearly five points above the originally reported one (92.99%) (Tables 12 and 10).""",
    """At matched operation count, global magnitude pruning retains 97.80% accuracy&mdash;about one point above the best corrected strategy at matched operations (96.88%, interpolated) and 4.81 points above the originally reported one (92.99%) (Tables 12 and 10).""",
)
rep(
    """but the compute benefit must be re-framed as an accuracy loss purchased at matched op counts, not a Pareto improvement.""",
    """but the compute benefit must be re-framed as an accuracy loss purchased at matched op counts, not a Pareto improvement. The same-mask control (C3a) sharpens this: the deficit is attributable to the zeroing mask, not to ternarizing survivors.""",
)

# ---- Conclusion ----
rep(
    """and matched-sparsity magnitude pruning retains 97.80% accuracy (a 0.01-point drop), dominating every ternary strategy's accuracy&ndash;op tradeoff on this benchmark. Nothing about the criterion survives attribution&mdash;not even its operating point: under corrected scaling the &tau; optimum moves to 0.2, and validation data held out from training concurs, so the original &tau;=0.3 selection was itself an artifact.""",
    """and matched-sparsity magnitude pruning retains 97.80% accuracy (a 0.01-point drop)&mdash;about one point above the best corrected ternary strategy at matched operations, with a same-mask control attributing the deficit to the zeroing mask rather than to ternarizing survivors (0.06 pp). The operating point is scaling-dependent: under corrected scaling the &tau; optimum moves to 0.2, and validation data held out from training concurs, though a five-point, single-seed grid does not resolve 0.2 versus 0.3.""",
)
rep(
    """We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but the demonstration that selection-criterion studies for partial quantization are unsafe without four controls: a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, a matched-operation pruning baseline, and training-holdout threshold selection under the deployed scaling.""",
    """We believe the most useful contribution of this paper is not the entropy criterion, which fails attribution, but what this case study suggests: selection-criterion studies for partial quantization are fragile without four controls&mdash;a sparsity-correlation diagnostic, a magnitude-restoring scaling condition, a matched-operation pruning baseline (with a same-mask decomposition), and training-holdout threshold selection under the deployed scaling.""",
)

open(P, "w", encoding="utf-8", newline="").write(t)
print("All HTML fixes applied OK")
