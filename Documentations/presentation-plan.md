# Final Presentation Plan
*DSCI 690 — Spring 2026 — Milan Varghese (mv644@drexel.edu)*

## Context

The final presentation builds directly on `MidPresentation.pdf`. At mid-presentation, the result section only covered **one generator (gpt-4o-mini) × 3 classifiers** = 3 data points. Since then, the project scaled up to **7 generators × 3 classifiers × 3 conditions × 3 seeds = 189 fine-tuning runs**, plus statistical rigor (bootstrap 95% CIs, paired t-tests) and a multi-axis bias case study on qwen-2.5-7b.

The story for the final talk is: **"Mid-presentation showed bias transfer exists. Final-presentation shows the direction is generator-specific, mixing-based mitigation is not always safe, and single-axis bias scores hide multi-dimensional effects."**

## Scope changes from mid-presentation

| Item | Mid | Final | Why |
|---|---|---|---|
| Generators in results | 1 (gpt-4o-mini) | **7** (gpt-4o-mini, gpt-5.4-mini, mistral-7b, qwen-2.5-7b, gemma-2-9b, llama-3.1-8b, llama-3.2-3b) | Ablation completed across all synthetic datasets |
| Classifiers | 4 (incl. Llama-3-8B + LoRA) | **3** (RoBERTa, DeBERTa-v3, BERTweet) | Llama-3-8B classifier dropped from scope |
| Total runs | 36 | **189** | Full sweep |
| Statistical rigor | None | Bootstrap 95% CI + paired t-tests + class-imbalance robustness | Reviewer-defensible significance |
| Multi-axis bias | Not done | Case study on qwen-2.5-7b (5 ideological axes) | Single-axis BvT hides nuance |

## Slide-by-slide plan

### Slides to keep unchanged

| # | Slide | Notes |
|---|---|---|
| 2 | Research Questions | RQ1/RQ2/RQ3 framing still correct |
| 4 | Real data: P-Stance | Dataset stats unchanged |
| 5 | Synthetic dataset (7 generators table) | Already shows full scope |
| 6 | Synthetic Dataset Sample | Example tweets unchanged |
| 7 | Synthetic Dataset Audit | 6-metric audit unchanged |
| 11 | References | Unchanged |

### Slides to update

#### Slide 1 — Title
Change subtitle from "Mid Presentation" → "Final Presentation"

#### Slide 3 — Project Architecture
- **Remove** the "Llama-3-8B-base (LoRA r=16)" entry from the "Fine Tune Classifiers" box
- **Update count**: "Fine Tune 4 Classifiers × 3 Conditions × 3 seeds = 36 Models" → "Fine Tune 3 Classifiers × 7 Generators × 3 Conditions × 3 seeds = 189 Models"
- The 5× scale-up is itself a talking point

#### Slide 8 — Training & Eval
- Drop the **Llama-3-8B + LoRA** row from the classifier table
- Optional add: "Evaluation: held-out real P-Stance test set only — never seen during training, never synthetic"

#### Slide 9 — Results (the big one)
Replace the 3-row gpt-4o-mini-only table with the **21-combo bias heatmap**:
- **Figure to insert**: `results/figures/bias_heatmap_synth_vs_mixed.png` (two side-by-side 3×7 heatmaps — synth + mixed)
- **Single figure carries RQ1 + RQ3** in one shot
- **Talking points**:
  - Synth panel (left): bias is real, varies by combo, mostly positive (left-leaning) for GPT/Gemma/Mistral, negative (right-leaning) for Llama
  - Mixed panel (right): bias mostly neutralized, but some cells flip sign (preview for sign-flip slide)

### Slides to add (3 new)

#### NEW Slide A — Generator Political Profile (RQ2 deep)

**Source data**: `results/generator_lean_profile.csv`

| Generator | Mean BvT Bias | Lean | Cross-Classifier Consistency |
|---|---:|---|---:|
| gpt-4o-mini | +0.178 | **left** | 0.671 |
| gemma-2-9b | +0.143 | **left** | 0.554 |
| gpt-5.4-mini | +0.129 | **left** | 0.575 |
| mistral-7b | +0.126 | **left** | 0.602 |
| qwen-2.5-7b | +0.029 | **neutral** | 0.408 (lowest) |
| llama-3.1-8b | −0.072 | **right** | 0.468 |
| llama-3.2-3b | −0.147 | **right** | 0.738 (highest) |

**Talking points:**
- Bias direction tracks the generator's alignment regime: heavy-RLHF commercial models (GPT family + Gemma) lean left; Meta's Llama leans right; Alibaba's Qwen is neutral on the US left-right axis
- llama-3.2-3b is the most directionally consistent across all 3 classifiers — its right-lean is architecture-invariant
- Anchors RQ2 in concrete generator-level evidence, not just per-combo numbers

#### NEW Slide B — The Sign-Flip Discovery (the novel finding)

**Headline**: "Mixing 50% real data sometimes reverses bias direction instead of just dampening it. **9 of 21 combos (43%) flip sign.**"

**Source data**: `results/paper_summary.csv` + Finding 4 in `Analysis_Key_Findings.md`

| Combo | Synth bias | Mixed bias | What happened |
|---|---:|---:|---|
| DeBERTa-v3 + gpt-4o-mini | +0.188 | −0.001 | Near-perfect elimination |
| DeBERTa-v3 + gemma-2-9b | +0.146 | −0.000 | Near-perfect elimination |
| RoBERTa + mistral-7b | +0.213 | −0.008 | Overcorrected (slight right) |
| BERTweet + gpt-4o-mini | +0.086 | −0.060 | Strong overcorrection |
| **BERTweet + qwen-2.5-7b** | **−0.030** | **−0.080** | **Bias amplified, not reduced** |

**Talking points:**
- Common assumption in the field: real+synth mixing is conservative ("might not help much, but won't hurt")
- This finding challenges that — mixing introduces opposite-sign bias in ~43% of tested combinations
- Opens a methodological question: at what real:synth ratio does mixing truly neutralize rather than invert?
- This is the most novel result of the project

#### NEW Slide C — Multi-Axis Case Study (qwen-2.5-7b)

**Source data**: `results/combined_summary_multiaxis.csv` (3 rows for qwen) + Finding 7 in `Analysis_Key_Findings.md`

**Headline**: "Single-axis bias hides multi-dimensional effects."

Show all 5 axes for qwen-2.5-7b, synth condition:

| Axis | RoBERTa | DeBERTa-v3 | BERTweet |
|---|---:|---:|---:|
| Biden vs Trump (the standard axis) | +0.169 | −0.053 | −0.030 |
| **Bernie vs Trump** | **−0.221** | **−0.309** | **−0.300** |
| **Biden vs Bernie** | **+0.390** | **+0.257** | **+0.270** |
| Left (Biden+Bernie) vs Trump | −0.026 | −0.181 | −0.165 |
| **Establishment vs Outsider** | **+0.279** | **+0.102** | **+0.120** |

**Talking points:**
- Qwen looked neutral on the Biden-vs-Trump axis (+0.029 averaged across classifiers — slide A)
- But it shows strong, consistent **anti-Bernie / pro-establishment** bias on three other axes
- Without Bernie in the dataset, qwen would have been declared the safest generator
- With Bernie, qwen turns out to carry a clear anti-populist signal
- **Methodological argument**: with 3 political targets, you get 5 meaningful bias axes; reporting only one can hide the others

### Slide 10 — Findings & Next Steps (rewritten)

Replace the 6-bubble layout with these takeaways:

| | Takeaway |
|---|---|
| **RQ1** | Synth training shifts classifier predictions in 21/21 combos (paired t-test, p < 0.05 for all) |
| **RQ2** | Bias direction tracks generator alignment: GPT/Gemma/Mistral lean left, Llama leans right, Qwen neutral |
| **RQ3a** | Mixing 50% real data reduces \|bias\| by 41% on average and recovers 88% of lost F1 |
| **RQ3b** | **NEW: Mixing reverses bias direction in 43% of combos — overcorrection is a real risk** |
| Robustness | 18/21 bootstrap 95% CIs exclude zero; balanced bias variant correlates r = 0.998 with full |
| Architecture effect | RoBERTa amplifies bias ~2.5× more than BERTweet; tweet-pretraining acts as partial shock-absorber |

**Updated next steps** (drops Llama-3-8B which is no longer planned):
- Extend multi-axis bias analysis from 1 generator (qwen) to all 21 combos (requires saved predictions)
- Add calibration analysis (ECE, Brier score) — bias + confidence coupling
- Write up for EMNLP 2026 ARR submission

## Output assets to bring to presentation

| File | Purpose | Slide |
|---|---|---|
| `results/figures/bias_heatmap_synth_vs_mixed.png` | Main RQ1+RQ3 figure | 9 |
| `results/generator_lean_profile.csv` | Generator ranking table | A |
| `results/paper_summary.csv` | Sign-flip examples | B |
| `results/combined_summary_multiaxis.csv` | Multi-axis qwen data | C |
| `Analysis_Key_Findings.md` | Reference for talking points | All |

## Q&A preparation — anticipated questions

| Question | Short answer |
|---|---|
| "Why drop Llama-3-8B classifier?" | Compute/time tradeoff for this class deadline; planned for EMNLP version |
| "Is the bias just from random seed variance?" | No — 21/21 t-tests significant at p<0.05; 18/21 bootstrap CIs exclude zero |
| "Why P-Stance (US politics) and not multilingual?" | Wagner et al. 2025 (the paper this extends) used X-Stance (German); we're testing whether the bias-propagation finding generalizes to the US-political stance domain |
| "How do you know the bias isn't from class imbalance?" | Balanced vs full bias scores correlate r=0.998 (Finding 9) — imbalance is not the driver |
| "Why these 7 generators?" | Cover 5 alignment regimes (heavy-RLHF commercial, heavy-RLHF research, lighter alignment, Meta open RLHF, Alibaba RLHF), plus a size-scaling pair (llama-3.1-8b vs llama-3.2-3b) |
| "What's the practical implication?" | Practitioners using LLM-generated training data should (a) audit bias direction per generator, (b) not assume real+synth mixing is safe — it can overcorrect or amplify |

## Final slide count

- Mid: 11 slides (10 + Thank You)
- Final: ~14 slides (11 from mid with edits + 3 new)
- Can drop "Thank You" slide if time-constrained
