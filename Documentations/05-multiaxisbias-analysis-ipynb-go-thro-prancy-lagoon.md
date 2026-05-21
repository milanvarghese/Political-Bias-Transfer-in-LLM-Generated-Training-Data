# Plan: Validate & Extend 05_MultiAxisBias_Analysis.ipynb → New 06_Veda_MultiAxisBias_Analysis.ipynb

## Context

`05_MultiAxisBias_Analysis.ipynb` is the primary analysis artifact for the EMNLP 2026 paper on political bias transfer in LLM-generated training data. It currently produces 7 figures and 2 CSVs but only implements a fraction of the 3-axis bias framework in `PROJECT_CONTEXT.md`. Despite being titled "MultiAxisBias," it only uses the Biden-vs-Trump axis (1 of 5). The other 4 axes exist in `results/combined_summary_multiaxis.csv` (for qwen-2.5-7b only) but are never loaded. Calibration analysis (Axis b), statistical tests, and the full class-distribution-shift table are all absent — each required for paper claims.

**Decision:** Preserve `05` unchanged. Create `06_Veda_MultiAxisBias_Analysis.ipynb` as a rigorous extension.

---

## Validation Findings

### What's Correct in 05
- BvT Directional Bias Score formula in the CSV matches `PROJECT_CONTEXT.md` exactly
- Three conditions (real / synth / mixed), 3 seeds (42, 123, 7), held-out real test set — all correct
- 7 generators × 3 classifiers → 21-row `combined_summary.csv` — consistent with CLAUDE.md
- `paper_summary_perseed.csv` is generated and includes std dev across seeds

### What's Present in Results but NOT Used in 05
- `results/combined_summary_multiaxis.csv` — 5 bias axes × `_full`/`_balanced` variants — **21 columns, only 3 rows** (qwen-2.5-7b × 3 classifiers)
- `results/bias_heatmap_5axis_synth.png` and `bias_heatmap_5axis_mixed.png` — generated outside the notebook

### Gaps vs Project Documentation

| # | Gap | Severity | Source |
|---|-----|----------|--------|
| 1 | No multi-axis visualization (5 axes exist for qwen, partial) | **Critical** | PROJECT_CONTEXT.md §Bias Axes |
| 2 | No calibration analysis (ECE + Brier) — Axis (b) entirely absent | **Critical** | PROJECT_CONTEXT.md Layer 2b |
| 3 | No Bootstrap 95% CI on bias scores or F1 | **Critical** | PROJECT_CONTEXT.md Layer 3 |
| 4 | No class distribution shift Δ(t,c) — only ΔF1 shown | **High** | PROJECT_CONTEXT.md Layer 2a |
| 5 | No McNemar's test for paired classifier comparison | **High** | PROJECT_CONTEXT.md Layer 3 |
| 6 | No cross-generator political profile ranking/correlation | **High** | Core research question |
| 7 | No input validation — silent failures on bad CSV shape | **Medium** | Bug risk |
| 8 | `combined_summary_multiaxis.csv` has only 3 of 21 expected rows | **Medium** | Data completeness |
| 9 | `_full` vs `_balanced` variants never explained or compared | **Low** | Clarity |
| 10 | Hardcoded `vmax=0.3/0.25` won't adapt if values exceed range | **Low** | Robustness |

### Inconsistencies with Original Proposal
- Original locked scope: 2 classifiers (RoBERTa + Llama-3-base). Actual: RoBERTa, DeBERTaV3, BERTweet. Llama-3-base classifier is missing from results — add a scope-change note.
- Original 2-generator hypothesis is now a 7-generator sweep — strengthens claims; frame this explicitly.

---

## Implementation Plan

### Files
- **Do NOT modify** [05_MultiAxisBias_Analysis.ipynb](05_MultiAxisBias_Analysis.ipynb) — preserve as baseline
- **Create new** `06_Veda_MultiAxisBias_Analysis.ipynb`

### Data files available (read-only)
- [results/combined_summary.csv](results/combined_summary.csv) — 21 rows, 14 cols
- [results/combined_summary_multiaxis.csv](results/combined_summary_multiaxis.csv) — 3 rows, 21 cols
- `results/{clf}/{gen}/real+synth+mixed/test_metrics.csv` — per-seed metrics
- [results/paper_summary_perseed.csv](results/paper_summary_perseed.csv) — 65 rows, mean/std per (clf, gen, condition)

---

## `06_Veda_MultiAxisBias_Analysis.ipynb` — Section-by-Section

### Section 0 — Header & Scope Note
Markdown cell: explain what this notebook adds over `05`, list all outputs, document scope expansion (2→7 generators, 2→3 classifiers, Llama-3-base classifier still pending).

### Section 1 — Imports + Constants
Same constants as `05` (TARGETS, CONDITIONS, SEEDS, CLF_ORDER, GEN_ORDER). Add `from scipy import stats` and `from sklearn.calibration import calibration_curve`.

### Section 2 — Data Loading + Validation
- Load `combined_summary.csv`; assert 21 rows, no NaN in key columns, all expected (clf, gen) combos present
- Load `combined_summary_multiaxis.csv`; print shape with note about 3-row partial coverage
- Load all per-seed `test_metrics.csv` files (same as `05`)
- Inspect column list of `test_metrics.csv` to determine if per-class prediction counts are present

### Section 3 — Reproduced Baseline (from 05)
Reproduce the 7 figures from `05` exactly (same code, same outputs). This makes `06` self-contained and confirms reproducibility.  
Outputs: same 7 PNGs as `05`, saved to `results/figures/06_*` prefix.

### Section 4 — Bias Score Variants: `_full` vs `_balanced`
Markdown explanation of both:
- `_full`: raw prediction counts (imbalance-sensitive)
- `_balanced`: equal weight per (target, class) cell  
4-panel heatmap: synth_full / synth_balanced / mixed_full / mixed_balanced.  
Output: `06_bias_heatmap_4panel.png`

### Section 5 — Bootstrap 95% CI on Bias Scores
Function `bootstrap_ci(values, n=2000, seed=42)` using `np.percentile` on resampled column means.  
Apply to `bias_synth_full` and `bias_mixed_full` per (clf, gen) row.  
Error bar plot: all 21 (clf, gen) combos on y-axis, CI bars on x-axis.  
Mark CIs that exclude zero (statistically significant bias).  
Save updated table as `results/paper_summary_with_ci.csv`.  
Output: `06_bias_ci_errorbar.png`

### Section 6 — Multi-Axis Bias (5 axes, qwen-2.5-7b)
Load `combined_summary_multiaxis.csv` (3 rows).  
For each of the 3 classifiers (roberta, debertav3, bertweet), create a `5×2` heatmap:
- Rows: B_vs_T, Ber_vs_T, B_vs_Ber, Left_vs_T, Estab_vs_Outsider
- Columns: synth vs mixed  
Shared `vmax = max(0.3, data.abs().max().max() * 1.1)`.  
Add bold note: "Full 21-row analysis requires re-running fine-tuning notebooks with raw prediction export."  
Output: `06_bias_5axis_qwen.png`

### Section 7 — Class Distribution Shift Δ(t,c)
Check `test_metrics.csv` for per-target per-class prediction columns (e.g., `trump_favor_pred_count`, `trump_against_pred_count`).  
If present: compute `Δ(t, c) = P_synth(ŷ=c|t) - P_real(ŷ=c|t)` for all 3 targets × 2 classes (6 values per row).  
Display as a 6-column heatmap per (clf, gen).  
Save as `results/delta_class_distribution.csv`.  
If absent: display the formula, show a data availability note, and document exactly which columns to add in the fine-tuning notebooks to enable this analysis.  
Output: `06_delta_class_distribution.png` (or data-gap documentation cell)

### Section 8 — McNemar's Test
Per (clf, gen) pair, compare "synth vs real" using per-seed F1 with a paired t-test (3 seeds, proxy for McNemar when raw predictions unavailable).  
If per-prediction data is available: build contingency table and use `scipy.stats.chi2_contingency`.  
Report p-values in a table; flag pairs where p < 0.05.  
Output: `06_mcnemar_pvals.csv`

### Section 9 — Cross-Generator Political Profile Ranking
Group `combined_summary.csv` by generator; compute:
- `mean_bias_synth` (signed mean across 3 classifiers)
- `std_bias_synth` (std across 3 classifiers)
- `lean_direction`: "left" if mean > 0.05, "right" if mean < -0.05, "neutral" otherwise
- `spearman_rho`: Spearman ρ of generator's bias vector (3 values) vs a reference ordering — measures cross-classifier consistency  
Horizontal bar chart sorted by `mean_bias_synth`, bars colored by lean direction (blue=left, red=right, gray=neutral), error bars = std.  
Save as `results/generator_lean_profile.csv`.  
Output: `06_generator_lean_ranking.png`

### Section 10 — Calibration Analysis (Axis b)
If logit/probability columns exist in `test_metrics.csv`: compute ECE (15-bin) and Brier score per (clf, gen, condition).  
If not: add a documented note explaining what probability output format is needed from fine-tuning notebooks, and provide the ECE/Brier formula as a reference.  
Grouped bar chart: ECE per condition (real / synth / mixed) per classifier.  
Output: `06_calibration_ece_brier.png` (or gap documentation)

### Section 11 — Summary & Paper Mapping
Markdown table mapping each notebook output to its corresponding paper table/figure number.  
List all CSVs and PNGs saved.  
Note remaining gaps (multi-axis for all 21 combos, Llama-3-base classifier, raw predictions for Δ(t,c) and ECE).

---

## New Outputs from `06`

| File | Content |
|------|---------|
| `results/paper_summary_with_ci.csv` | 21 rows + bootstrap CI columns for bias scores |
| `results/delta_class_distribution.csv` | Δ(t,c) table (conditional on data) |
| `results/generator_lean_profile.csv` | Generator ranking by signed bias + lean label |
| `results/06_mcnemar_pvals.csv` | p-values for synth-vs-real comparison |
| `results/figures/06_bias_heatmap_4panel.png` | `_full` vs `_balanced` side-by-side |
| `results/figures/06_bias_ci_errorbar.png` | Bootstrap CI error bars |
| `results/figures/06_bias_5axis_qwen.png` | 5-axis heatmap for qwen-2.5-7b |
| `results/figures/06_delta_class_distribution.png` | Δ(t,c) heatmap |
| `results/figures/06_generator_lean_ranking.png` | Signed-bias generator ranking |
| `results/figures/06_calibration_ece_brier.png` | ECE/Brier per condition |

---

## Verification

1. Run all cells top-to-bottom — no errors; all 10 new figures render
2. `paper_summary_with_ci.csv`: 21 rows, 4 CI columns, no NaN — verify 1-2 CIs by hand
3. `combined_summary_multiaxis.csv` loads and prints "3 rows × 21 cols" with partial coverage note
4. Spot-check bias score formula: take any row's `trump_synth_dF1` + `biden_synth_dF1` values, apply `PROJECT_CONTEXT.md` formula manually, confirm it matches `bias_synth_full`
5. Generator lean ranking: GPT-family (gpt-4o-mini, gpt-5.4-mini) should show positive/left-lean; open-source models show varied directions
6. All `06_*` figures saved under `results/figures/` without overwriting any `05` outputs
