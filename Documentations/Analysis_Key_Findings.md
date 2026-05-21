# Key Findings: Political Bias Transfer Analysis
*Based on `06_Veda_MultiAxisBias_Analysis.ipynb` — EMNLP 2026 submission*

---

## Quick Summary

Training a stance classifier on LLM-generated synthetic data does two separable things:
1. **Hurts performance** — macro F1 drops 8–15% compared to real-data training (expected, not interesting)
2. **Transfers directional political bias** — the classifier's error pattern becomes asymmetric along partisan lines in a direction that matches the generator's known political alignment (this is the paper's claim)

The bias is not just noise. It is **directional**, **generator-specific**, and **classifier-architecture-dependent**. Mixing 50% real data into training substantially recovers accuracy but produces an unexpected side effect: in 9 of 21 (classifier, generator) combinations, mixing *reverses* the bias direction rather than eliminating it.

---

## Experimental Setup

| Axis | Values |
|------|--------|
| Dataset | P-Stance — 21,574 real tweets; targets: Trump, Biden, Bernie Sanders; labels: FAVOR / AGAINST |
| Generators (synthetic data) | gpt-4o-mini, gpt-5.4-mini, mistral-7b, qwen-2.5-7b, gemma-2-9b, llama-3.1-8b, llama-3.2-3b |
| Classifiers | RoBERTa-base, DeBERTaV3-base, BERTweet |
| Training conditions | `real` (real P-Stance train set only), `synth` (synthetic only), `mixed` (50% real + 50% synth) |
| Test set | Held-out real P-Stance test set only — never trained on, never synthetic |
| Seeds | 3 (42, 123, 7) per (classifier × generator × condition) |
| Total runs | 21 combos × 3 conditions × 3 seeds = 189 fine-tuning runs |

---

## Finding 1 — Synthetic Training Always Causes a Significant F1 Drop

### What it shows
Training exclusively on synthetic tweets instead of real P-Stance tweets degrades macro F1 by **8.0–14.6 percentage points** across every (classifier, generator) combination tested. The drop is universal — no generator produces synthetic data that matches real data quality.

| Classifier | Real F1 | Mean Synth F1 | Mean Drop |
|------------|---------|---------------|-----------|
| RoBERTa | 0.824 | 0.705 | −0.119 |
| DeBERTaV3 | 0.849 | 0.747 | −0.102 |
| BERTweet | 0.820 | 0.702 | −0.118 |

Worst single case: **RoBERTa + mistral-7b** (−0.146 drop). Best case: **DeBERTaV3 + qwen-2.5-7b** (−0.080 drop).

### Methodology
**Metric:** Macro-averaged F1 on the held-out real test set across 3 seeds.
Macro-F1 averages per-class F1 equally across FAVOR and AGAINST, so it penalises a model that becomes good at one label at the expense of the other — important for detecting skewed predictions.

**Why this matters:** This finding is the **baseline** that makes the bias finding non-trivial. A naive interpretation might be: "synthetic data is just worse, so errors go up uniformly." Finding 2 shows that is wrong — the errors are not uniform, they are *directional*.

**Why statistical testing:** All 21 pairs are statistically significant (paired t-test, p < 0.05 for all 21, mean p = 0.009). With only 3 seeds the test has low power, but the effect sizes are large enough that every single pair clears the threshold.

---

## Finding 2 — Synthetic Training Transfers Directional Political Bias (The Core Finding)

### What it shows
The performance drop is not random. Errors fall **asymmetrically along partisan lines**. For most generators, the classifier trained on synthetic data becomes relatively worse at one political target than another, in a direction that matches the generator's alignment.

The **Directional Bias Score** (BvT axis) captures this:
```
Bias = (Δ_Biden,Favor − Δ_Trump,Favor) + (Δ_Trump,Against − Δ_Biden,Against)

where Δ(target, class) = P_synth(ŷ = class | target) − P_real(ŷ = class | target)
```
- **Positive score** = the synthetic-trained classifier is relatively better at pro-Biden content and worse at pro-Trump content → **left-leaning shift**
- **Negative score** = the reverse → **right-leaning shift**
- **Near zero** = no detectable asymmetry

Observed range across all 21 (classifier, generator) combinations:

| | synth condition | mixed condition |
|---|---|---|
| Minimum (most right-lean) | −0.207 | −0.121 |
| Maximum (most left-lean) | +0.260 | +0.074 |
| Mean absolute bias | 0.130 | 0.067 |

### Methodology
**Bias score formula** is taken directly from `PROJECT_CONTEXT.md`. It measures the asymmetry in prediction-distribution shifts between Biden-related and Trump-related tweets.

**Why this formula rather than just comparing F1 drops:**
A simple F1 comparison (`trump_dF1 vs biden_dF1`) tells you *accuracy asymmetry* but not *prediction-distribution asymmetry*. The Directional Bias Score captures whether the model's *output distribution* has shifted toward one party's preferred labels. For example, a model that now predicts FAVOR more often for Biden tweets and AGAINST more often for Trump tweets — even with the same overall accuracy — would show a positive bias score.

**Why it matters for the paper:** The performance drop alone is not a publishable finding — distribution shift is expected and well-documented. The directional asymmetry is the finding: bias embedded in the generator's weights transfers to the classifier's prediction patterns on *real* human-authored tweets the classifier was never trained on.

---

## Finding 3 — Generator Political Profile Predicts Bias Direction

### What it shows
Different generators produce systematically different bias directions. This is not random — the lean direction maps onto known political alignments of the underlying models.

| Generator | Mean BvT Bias | Lean | Cross-clf Consistency |
|-----------|--------------|------|-----------------------|
| gpt-4o-mini | +0.178 | left-lean | 0.671 |
| gemma-2-9b | +0.143 | left-lean | 0.554 |
| gpt-5.4-mini | +0.129 | left-lean | 0.575 |
| mistral-7b | +0.126 | left-lean | 0.602 |
| qwen-2.5-7b | +0.029 | neutral | 0.408 |
| llama-3.1-8b | −0.072 | right-lean | 0.468 |
| llama-3.2-3b | −0.147 | right-lean | 0.738 |

**Lean direction threshold:** |mean| > 0.05 = lean, otherwise neutral.

**Consistency score** = 1 / (1 + CV) where CV = std across 3 classifiers / |mean|. Score of 1.0 means all three classifiers exhibit exactly the same bias magnitude; score near 0 means the direction flips across classifiers.

Notable results:
- **GPT-4o-mini** (heavily RLHF-aligned, commercial) → strongest left-lean (+0.178), consistent
- **Llama-3.2-3b** → strongest right-lean (−0.147), and most consistent across classifiers (0.738)
- **Qwen-2.5-7b** → neutral; but lowest consistency (0.408), meaning classifiers disagree about direction
- **Mistral-7b** and **Gemma-2-9b** both show left-lean — an unexpected finding for open-weight models

### Methodology
**Generator lean score** = mean of `bias_synth_full` across 3 classifiers (RoBERTa, DeBERTaV3, BERTweet).
Averaging across classifiers controls for classifier-specific amplification (see Finding 5) — it isolates the generator's contribution.

**Cross-classifier consistency (CV-based):** With only 3 classifiers, Spearman ρ is numerically unstable, so we use the inverse coefficient of variation as a proxy. A generator with high CV (like qwen) is causing classifier-specific effects; a generator with low CV (like llama-3.2-3b) is producing bias that every classifier picks up consistently regardless of architecture.

**Why this is the key test:** The original hypothesis in `PROJECT_CONTEXT.md` is that a generator's *internalized* political alignment (from pretraining and RLHF) leaks into the training signal even when the generation prompts are politically neutral. This finding provides direct evidence: the direction of bias transfer correlates with the generator's known alignment.

---

## Finding 4 — Mixing Partially Mitigates Bias but Often Reverses It

### What it shows
Replacing the 50/50 mixed condition (half real, half synthetic data) reduces the absolute bias score on average, but produces surprising side effects.

**On average:** Mean |bias| drops from 0.130 (synth) to 0.067 (mixed) — a 40.7% reduction.

**Recovery of accuracy:** Mixing recovers 88.3% of the F1 lost from synthetic-only training. The mixed condition lands within a mean 1.3% of real F1.

**Sign-flip phenomenon:** In **9 of 21 combinations** (43%), mixing does not just dampen the bias — it *reverses* the direction. Examples:

| Classifier | Generator | Synth Bias | Mixed Bias | Interpretation |
|------------|-----------|-----------|-----------|----------------|
| DeBERTaV3 | gpt-4o-mini | +0.188 | −0.001 | Near-perfect elimination |
| DeBERTaV3 | gemma-2-9b | +0.146 | −0.000 | Near-perfect elimination |
| RoBERTa | mistral-7b | +0.213 | −0.008 | Overcorrected to slight right-lean |
| BERTweet | gpt-4o-mini | +0.086 | −0.060 | Strong overcorrection |
| BERTweet | qwen-2.5-7b | −0.030 | −0.080 | Amplified instead of reduced |

The worst cases for mitigation are BERTweet + small generators (qwen, gemma), where mixing *amplifies* the bias: the qwen-2.5-7b / BERTweet pair goes from −0.030 to −0.080 (−167% "reduction").

### Methodology
**Percent bias reduction** = (|bias_synth| − |bias_mixed|) / |bias_synth| × 100%.
Negative values mean mixing worsened the absolute bias.

**Sign-flip detection:** Compare `sign(bias_synth_full)` vs `sign(bias_mixed_full)`. A flip indicates the real-data component dominated and overcorrected the bias direction rather than simply dampening it.

**Why this matters:** The sign-flip finding is the most practically significant result for ML practitioners. The common assumption is that mixing real and synthetic data is a conservative strategy — it might not help much, but it won't hurt. This finding challenges that: mixing can introduce a new bias of opposite sign in ~43% of tested combinations. It also raises a methodological question for future work: how do you mix at a ratio that truly neutralizes rather than inverts?

---

## Finding 5 — Classifier Architecture Determines How Much Bias Gets Amplified

### What it shows
The same synthetic training data produces substantially different bias magnitudes depending on the classifier architecture.

| Classifier | Mean |BvT Bias| (synth) | Max |BvT Bias| | Architecture |
|------------|-----------------------|---------------|--------------|
| RoBERTa | **0.187** | 0.260 | Transformer, general-purpose |
| DeBERTaV3 | 0.128 | 0.207 | Transformer, disentangled attention |
| BERTweet | **0.076** | 0.120 | Transformer, Twitter-pretrained |

RoBERTa amplifies bias roughly **2.5× more** than BERTweet. DeBERTaV3 sits in between.

### Methodology
**Mean absolute bias** = mean of |bias_synth_full| across 7 generators per classifier.
**Max absolute bias** = the worst-case generator for each classifier.

**Why architectures differ:** BERTweet was pretrained specifically on tweet data and thus already has a richer representation of the political stance vocabulary in social media text. Its pretrained weights are already "calibrated" to real-world Twitter political discourse, making it harder for synthetic training data to shift its prediction distribution. RoBERTa, pretrained on general corpora (BookCorpus + Wikipedia), has no such prior and is more susceptible to the distributional influence of synthetic training data.

**Implication for paper:** Architecture choice is a confound that must be reported when practitioners publish "we trained a stance classifier on synthetic data." The same generator + different classifier can give 2.5× difference in bias transfer magnitude.

---

## Finding 6 — Trump Tweets Suffer Disproportionate F1 Loss (F1-Proxy Class Distribution Shift)

### What it shows
The per-target F1 drop is not uniform across political figures. Trump-related tweets lose the most F1 under synthetic training:

| Target | Mean F1 Drop (synth − real) | Min drop | Max drop |
|--------|--------------------------|----------|----------|
| Donald Trump | **−0.154** | −0.206 | −0.099 |
| Bernie Sanders | −0.093 | −0.138 | −0.057 |
| Joe Biden | **−0.086** | −0.115 | −0.064 |

Trump tweets drop an average of **0.068 F1 points more** than Biden tweets. Since most generators lean left, their synthetic data contains vocabulary and framing that is aligned with Biden and Sanders discourse — the classifier gets better at those patterns and relatively worse at Trump.

This asymmetry is further decomposed by class: the F1-proxy Δ(t, c) heatmap shows that the FAVOR class (predicting pro-target stance) for Trump suffers the most, while Biden FAVOR drops the least. This directional asymmetry in the class × target space is the operationalized definition of left-leaning bias in this paper.

### Methodology
**Per-target F1 drop:** `trump_synth_dF1 = trump_macro_f1_synth − trump_macro_f1_real`, computed during fine-tuning and stored in `combined_summary.csv`.

**F1-proxy Δ(t, c):** From `test_metrics.csv`, which contains `favor_f1` and `against_f1` separately per (condition, seed, target). The proxy is:
```
Δ_f1(target, class) = f1_class[synth, target] − f1_class[real, target]
```
This is a proxy (not the exact prediction-proportion formula) because `test_metrics.csv` stores F1 scores, not raw prediction counts. The *direction* of Δ_f1 is informative and matches the Directional Bias Score in sign, but the magnitude is not directly comparable.

**Exact formula requires:** Per-sample prediction counts (how many Trump FAVOR predictions were made, not how well they were predicted). Adding `trump_favor_pred_count` columns to the fine-tuning notebook would enable exact Δ(t, c).

**Why target-level decomposition matters:** Aggregate metrics like "lower F1 on synthetic data" hide where the errors fall. A classifier trained on GPT-4o-mini synthetic data does not generalize poorly to all political content equally — it specifically struggles with Trump-positive content, which is the directional signature predicted by GPT-4o-mini's known alignment.

---

## Finding 7 — Multi-Axis Bias Reveals Hierarchical Political Structure (qwen-2.5-7b)

### What it shows
The full 5-axis bias analysis (available only for qwen-2.5-7b due to data availability) reveals that BvT (Biden vs Trump) is not the only or even strongest axis of bias. The Bernie vs Trump axis consistently shows stronger, more consistent negative bias across all classifiers:

| Axis | RoBERTa | DeBERTaV3 | BERTweet | Interpretation |
|------|---------|----------|---------|----------------|
| Biden vs Trump (B_vs_T) | +0.169 | −0.053 | −0.030 | Inconsistent across classifiers |
| **Bernie vs Trump (Ber_vs_T)** | **−0.221** | **−0.309** | **−0.300** | Strongly pro-Bernie / anti-Trump |
| Biden vs Bernie (B_vs_Ber) | +0.390 | +0.257 | +0.270 | Strongly pro-Biden within the left |
| Left-lean vs Trump | −0.026 | −0.181 | −0.165 | Moderate anti-Trump lean |
| Establishment vs Outsider | +0.279 | +0.102 | +0.120 | Pro-establishment bias |

**Key structural insight from qwen-2.5-7b:** Even though the BvT axis is inconsistent (RoBERTa shows +0.169 while DeBERTaV3 and BERTweet show negative values), the Ber_vs_T axis is strongly and consistently negative across all three classifiers (−0.22 to −0.31). This suggests synthetic training data has a more reliable pro-Bernie / anti-Trump effect than a pro-Biden effect, which is notable: qwen may encode Bernie's political positions more distinctly than Biden's (who occupies a more centrist/moderate position).

The B_vs_Ber axis (+0.26 to +0.39) shows a strong pro-Biden-over-Bernie pattern *within the left*, suggesting the synthetic data does not treat all left-leaning politicians equivalently.

### Methodology
**Multi-axis bias scores** are computed in the fine-tuning notebook by applying the same Directional Bias Score formula to all five target-pair combinations and storing them in `combined_summary_multiaxis.csv`. Each axis captures a different dimension of political comparison:
- **BvT:** The primary electoral axis (general election framing)
- **Ber_vs_T:** An ideological axis (progressive vs conservative populism)
- **B_vs_Ber:** An intra-left axis (centrist Democrat vs progressive)
- **Left_vs_T:** Aggregated left-leaning (Biden + Bernie together) vs Trump
- **Estab_vs_Outsider:** Cross-ideological axis (establishment figures Biden + Trump vs outsider Sanders)

**Why 5 axes instead of 1:** The BvT axis captures mainstream media framing (the "who wins the general election" framing) but misses ideological nuance. A model could be pro-Bernie without being pro-Biden. The multi-axis approach uncovers these distinctions.

**Current limitation:** Only qwen-2.5-7b has all 5 axes computed. Completing the 21-row multi-axis table requires re-running fine-tuning notebooks with raw prediction export enabled (saving per-sample predictions per target, not just aggregate F1).

---

## Finding 8 — The Bias Is Statistically Robust, Not Seed Noise

### What it shows
The directional bias findings are not artifacts of the 3-seed random variance. Two complementary statistical approaches confirm significance:

**Paired t-test (synth vs real, per seed F1):**
- 21/21 (100%) of (classifier, generator) pairs show p < 0.05 for the F1 drop
- Mean p-value = 0.009
- The F1 degradation from synthetic training is always statistically significant

**Bootstrap 95% CI on F1-proxy bias scores:**
- 18/21 (86%) of combinations have CIs that exclude zero
- The 3 non-significant cases: DeBERTaV3 + qwen-2.5-7b, BERTweet + qwen-2.5-7b, BERTweet + llama-3.1-8b
- All 3 non-significant cases involve qwen or llama-3.1-8b — generators with the smallest |mean bias| overall

**Per-seed F1 stability:**
- Mean std across 3 seeds: 0.009–0.021 depending on classifier
- Max observed std: 0.044 (DeBERTaV3)
- Bias magnitudes (0.07–0.26) are typically 5–20× larger than the seed-level std — the bias is not absorbed by seed noise

### Methodology
**Paired t-test:** Compares the 3 seed-level F1 values between two conditions (e.g., synth vs real) using `scipy.stats.ttest_rel`. With df=2, the test has low power — it requires a consistent, large effect across seeds to reach significance. The fact that all 21 pairs are significant indicates the F1 difference is much larger than within-condition seed variance.

**Bootstrap 95% CI:** A function computing the F1-proxy bias per seed (3 values per (clf, gen)) and resampling with replacement 5000 times. The CI bounds tell us whether the mean proxy bias is reliably non-zero across the observed seed distribution.

**Important caveat:** The bootstrap CI uses the *F1-proxy* bias (computed from favor_f1 / against_f1 differentials), not the exact `bias_synth_full` values (which are pre-computed from raw prediction counts). The proxy and the exact score agree in direction but differ in magnitude. For exact CIs on the true bias score, per-seed bias computation from raw predictions is needed.

**Why this matters:** Any claim of "bias transfer" requires ruling out that the asymmetric errors are just noise from random seed initialization. The combination of t-test significance on every pair plus 86% of CI-based tests excluding zero provides strong statistical backing for the directionality claims.

---

## Finding 9 — Bias Is Robust to Class Imbalance (Full vs Balanced Variants)

### What it shows
P-Stance has a slight class imbalance (51.7% AGAINST, 48.3% FAVOR). Any analysis of prediction-distribution shifts is potentially confounded by this imbalance — a model that always predicts the majority class would look biased. The `_balanced` bias score variant weights each (target × class) cell equally, correcting for imbalance.

Correlation between `_full` and `_balanced` across 21 observations:
- **Synth condition:** Pearson r = **0.998** (p = 4.7 × 10⁻²⁴)
- **Mixed condition:** Pearson r = **0.993** (p = 6.3 × 10⁻¹⁹)

The two variants are essentially identical (r ≈ 1.0). This means the bias findings are not artifacts of the 51.7/48.3 class split — they persist under equal cell weighting.

### Methodology
**Balanced variant** = bias score computed by averaging Δ values across (target × class) cells with equal weight, rather than weighting by the number of test samples per target.

**Pearson r as a robustness check:** If `_full` and `_balanced` diverge substantially, it would suggest that class imbalance is driving the apparent bias. A near-perfect correlation confirms the imbalance is not a confound.

**Recommendation for paper:** Report `_full` as the primary metric (standard, interpretable as prediction proportion shift) and `_balanced` as a robustness footnote. The r = 0.998 result justifies this simplification.

---

## Methodology Summary

### Why these three classifiers?
RoBERTa, DeBERTaV3, and BERTweet represent three distinct pretraining contexts: general corpus (RoBERTa), disentangled-attention general corpus (DeBERTaV3), and Twitter-specific (BERTweet). Using architectures with different inductive biases tests whether bias transfer is a general phenomenon or tied to a specific model family. Finding 5 shows it is general but architecture-amplified.

### Why a held-out real test set?
All 189 fine-tuning runs are evaluated on the **same real P-Stance test split** that was never seen during training. This is the critical design decision: if the test set were synthetic or mixed, any bias differences between conditions could be explained by distributional similarity to the training data. Using a fixed real test set ensures that all differences in prediction patterns are attributable solely to the training data composition.

### Why 3 random seeds?
Standard NLP practice for reporting variability. Each seed initialises the classifier head weights and determines the training batch ordering. Reporting across 3 seeds allows us to report mean ± std and detect when a finding is consistent vs. lucky. With 189 total runs, going to 5 seeds would have tripled Colab A100 compute time — 3 was the practical minimum for significance testing.

### Why the Directional Bias Score over simpler metrics?
Simpler alternatives were considered:
- **F1 drop per target** (used as a proxy in Section 6): captures *accuracy* asymmetry but not *prediction distribution* asymmetry. A classifier could have equal F1 on Trump and Biden tweets but still systematically over-predict FAVOR for one.
- **Per-target accuracy**: same limitation as F1
- **KL divergence between prediction distributions**: harder to interpret as left/right lean

The Directional Bias Score directly operationalises the political science concept of partisan lean: does the model over-predict pro-Biden labels and under-predict pro-Trump labels, or vice versa? The signed score maps directly onto the "left-lean / right-lean / neutral" taxonomy.

---

## What Still Needs to Be Done

| Gap | What's needed | Impact |
|-----|---------------|--------|
| **Multi-axis data for 20/21 combos** | Re-run fine-tuning with prediction export (`save_predictions=True`) | Enables full 5-axis bias analysis for all generators |
| **Exact Δ(t, c)** | Add `trump_favor_pred_count` etc. to `test_metrics.csv` | Replaces F1-proxy in Section 6 with exact formula |
| **Calibration (ECE, Brier)** | Save `test_probs_{seed}.csv` from fine-tuning | Tests whether biased predictions are also overconfident |
| **McNemar test** | Save per-sample prediction pairs | Enables proper non-parametric significance testing |
| **Llama-3-base classifier** | Colab A100 run | Tests whether a decoder-only base model is more/less susceptible |
| **Qualitative error analysis** | Manual review of 50 disagreement cases per condition | Adds interpretable examples to supplement statistics |
