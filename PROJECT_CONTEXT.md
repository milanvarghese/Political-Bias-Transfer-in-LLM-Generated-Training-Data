# Project Context: Political Bias Transfer in LLM-Generated Training Data

---

## Quick Summary

**Title:** Measuring Political Bias Transfer from LLM-Generated Training Data to Stance Classifiers

**Course:** DSCI 690 — Modeling Natural Language (Spring 2026, Drexel University)

**Team:** Milan Varghese

**Timeline:** ~5 weeks (class deliverable), targeting EMNLP 2026 ARR submission (May 25, 2026)

**One-line pitch:** We test whether the political bias of an LLM used to generate synthetic training data silently transfers to classifiers trained on that data.

**Scope status:** ✅ Locked at Version A (2 generators: GPT-4o-mini + Llama-3-8B-Instruct). Stretch goal of adding GPT-5.4-mini or Claude Haiku 4.5 only if ahead of schedule by end of Week 2.

---

## Core Research Question

When GPT-4o-mini or Llama-3-8B-Instruct is used to generate synthetic training data for a political stance classifier, does the generator's known political bias silently transfer to the classifier's predictions on real social media text?

---

## Three Specific Research Questions

1. Does training on GPT-4o-mini synthetic tweets shift classifier predictions on real tweets compared to training on real data?
2. Does Llama-3-Instruct synthetic data produce a different shift than GPT-4o-mini, matching each model's known political lean?
3. Does mixing real and synthetic data reduce the shift?

---

## Why This Matters

- LLMs are increasingly used to generate synthetic training data when labeled data is scarce.
- Wagner et al. (ICLR 2025 Spotlight) showed this works well for stance detection performance but explicitly flagged bias transfer as an open concern.
- LLMs have documented political biases (OpenAI models lean left, Llama-3 has a different profile).
- No one has empirically tested whether the generator's political bias transfers through synthetic data to the downstream classifier.
- Directly relevant to EMNLP 2026's Special Theme: "Data as a bottleneck and a responsibility" including "consequences of synthetic data feedback loops."

---

## Dataset: P-Stance

- **Source:** https://github.com/chuchun8/PStance
- **Size:** 21,574 real tweets
- **Targets:** Donald Trump, Joe Biden, Bernie Sanders
- **Labels:** **Binary — Favor / Against** (no Neutral class)
- **Class balance:**
  - Overall: 51.7% Against / 48.3% Favor
  - Per target: Bernie skews Favor (~56%), Biden and Trump skew Against (~54-56%)
  - Train/val/test stratification preserves the ~52/48 ratio
- **Source:** Li et al. 2021, Findings of ACL-IJCNLP

---

## Locked Scope: Version A

### Generators (2)
| # | Model | Role | Cost |
|---|---|---|---|
| 1 | **GPT-4o-mini** (OpenAI API, Batch) | Heavily aligned commercial (RLHF) | ~$0.30–0.50 |
| 2 | **Llama-3-8B-Instruct** (HuggingFace) | Lighter aligned, open-source | Free (run on Drexel cluster) |

**Note on model choice:** We use `gpt-4o-mini` rather than newer GPT-5.x models because (a) it represents the alignment regime currently in widespread practitioner use for synthetic-data pipelines, (b) it is the model most extensively studied in the political-bias-of-LLMs literature (Santurkar 2023, Röttger 2024), and (c) its long-term API stability supports replication. Newer reasoning models (GPT-5.x family) are flagged for future work / stretch comparator. GPT-4o (full) was originally planned; we use the cheaper `mini` variant since the bias-transfer signal does not depend on model size.

### Classifiers (2)
| # | Model | Method | Hub ID |
|---|---|---|---|
| 1 | **RoBERTa-base** (125M) | Full fine-tuning | `FacebookAI/roberta-base` |
| 2 | **Llama-3-8B** (base, NOT Instruct) | LoRA (r=16) | `meta-llama/Meta-Llama-3-8B` |

**Important:** Llama-3-8B-**Instruct** is used for *generation*; Llama-3-8B-**base** is used for *classification fine-tuning*. They are different model variants.

### Training Conditions (4)
| # | Condition | Training Data |
|---|---|---|
| 1 | Real only | P-Stance training split |
| 2 | GPT-4o-mini only | GPT-4o-mini synthetic data |
| 3 | Llama-3 only | Llama-3-Instruct synthetic data |
| 4 | Mixed | 50% real + 50% GPT-4o-mini synthetic |

### Total Fine-tuning Runs
- **8 base runs** (2 classifiers × 4 conditions)
- **24 runs with 3 seeds** for statistical significance
- Total compute: ~18 GPU hours on H100

---

## Methodology

### Step 1: Generate Synthetic Data
- Use GPT-4o-mini and Llama-3-8B-Instruct to generate ~7,200 tweets each
- Balanced across 6 target × stance combinations (3 targets × 2 stances)
- Neutral prompts — do NOT inject political framing into the prompt
- Use 3–5 prompt variations per stance to test phrasing sensitivity (optional if time-tight)
- Each prompt specifies target + stance; label is recorded automatically at generation time
- No manual annotation required
- Track refusal rate per (target × stance) cell — asymmetric refusals are themselves a finding

### Step 1.5: Synthetic Data Audit
Before training any classifier, audit generated synthetic tweets for:
- **Political lean of synthetic content:** Run lexicon-based partisan word scoring (e.g., AFINN, Connotation Frames) and a pretrained political stance classifier on the generated tweets themselves
- **Memorization / data leakage check:** Compute n-gram overlap (5-gram, 7-gram) and embedding similarity (sentence-BERT cosine) between synthetic tweets and real P-Stance training tweets
- **Quality filter:** Remove duplicates, malformed outputs, refusals
- This audit produces a per-condition characterization of the synthetic data BEFORE we ask "does it bias the classifier?"

### Step 2: Fine-Tune Classifiers
8 base runs (24 with 3 seeds) across the 4 training conditions × 2 classifier architectures.

### Step 3: Evaluation
- **ALL classifiers tested on the same held-out real P-Stance test set — never on synthetic data**
- This ensures any prediction differences are attributable only to training data

### Step 4: Measurement (3 layers)

**Layer 1 — Standard performance:**
- Macro-averaged F1
- Per-target F1 (Trump, Biden, Sanders)
- Precision and Recall per class

**Layer 2 — Bias shift, formally defined along three axes:**

**(a) Class Distribution Shift:**
For each target *t* and class *c*:
Delta(t, c) = P_synthetic(yhat = c | target = t) - P_real(yhat = c | target = t)

**(b) Calibration Shift:**
- Expected Calibration Error (ECE) per condition
- Brier score per condition
- Tests whether classifiers predict same labels but with systematically different confidence on certain political targets

**(c) Directional Bias Score:**
Bias Score = (Delta_Biden,Favor - Delta_Trump,Favor) + (Delta_Trump,Against - Delta_Biden,Against)
- Positive = left-leaning shift
- Negative = right-leaning shift
- Near zero = no detectable bias transfer

**Statistical rigor:**
- McNemar's test for paired prediction differences between classifiers
- Bootstrap 95% confidence intervals on all shift metrics
- 3 random seeds per training run

**Layer 3 — Error analysis & model comparison:**
- Disagreement rate between RoBERTa and Llama-3 classifiers under same condition
- Per-target confusion matrix differences
- Confidence-under-bias comparison: do larger models exhibit more confident biased predictions?
- Qualitative analysis of 50 cherry-picked disagreement cases per condition

---

## Critical Conceptual Point

**The performance drop is expected. The directional asymmetry in that drop is the finding.**

A classifier trained on synthetic data will lose some accuracy on real tweets due to distribution shift. That alone is not interesting. The research question is whether errors fall asymmetrically along political lines — e.g., does a GPT-4o-mini-trained classifier systematically miss pro-Trump tweets more than pro-Biden tweets, compared to the real-data baseline?

That asymmetry is what constitutes bias transfer.

---

## On the "Neutral Prompts" Assumption

The use of neutral prompts is intentional and central to the study. The hypothesis is that even **with neutral prompting**, an LLM's internalized political bias leaks into the generated text — because the model is generating from its own learned distributions over political language.

We do not assume neutral prompts produce neutral output. We hypothesize the opposite, and the synthetic data audit (Step 1.5) is designed to test this directly. If the audit shows GPT-4o-mini's "Favor Trump" tweets contain more hedging language, weaker emotional intensity, or different lexical choices than real "Favor Trump" tweets, that is itself a finding — and it sets up the bias transfer measurement that follows.

In the paper: *"We deliberately use minimal, label-only prompts to simulate naive practitioner usage. Any bias in the resulting data reflects the LLM's internalized priors, not researcher intent."*

---

## On Data Leakage

**The concern:** GPT-4o-mini and Llama-3 likely saw P-Stance or similar political tweets during pretraining. Synthetic tweets may inadvertently regurgitate or paraphrase real training tweets, blurring the line between "real" and "synthetic" data.

**How we address it:**

1. **Quantify overlap.** Compute n-gram (5-gram, 7-gram) and sentence-embedding cosine similarity between generated synthetic tweets and the real P-Stance training split. Report % of synthetic tweets that are near-duplicates.
2. **Frame the study around realistic deployment.** Real practitioners using LLMs for synthetic data face the exact same leakage problem.
3. **Optional leakage-filtered condition.** If overlap is non-trivial, run an additional training condition with near-duplicates removed.

---

## Limitations

- **P-Stance label assumption:** Human annotators carry implicit biases. Inter-annotator agreement (Krippendorff's alpha around 0.7) is a ceiling on what we can measure.
- **Dataset generalization:** P-Stance covers only 3 candidates from the 2020 U.S. election cycle. Findings may not generalize to other political contexts (non-U.S., non-electoral, non-Twitter) or non-political stance tasks.
- **Generator selection:** Only two LLMs are tested. Other models (Claude, Gemini, Mistral, Grok, newer GPT-5.x family) may exhibit different bias profiles. Future work.
- **Temporal validity:** Both LLMs and political discourse shift over time. Findings reflect a specific snapshot.
- **Prompt sensitivity:** Results may depend on the specific phrasing of "neutral" prompts.
- **Data leakage:** Acknowledged and addressed via overlap measurement and optional leakage-filtered ablation.

---

## Model Choice Justification

**Why GPT-4o-mini vs. Llama-3-8B-Instruct?**

| Dimension | GPT-4o-mini | Llama-3-8B-Instruct |
|---|---|---|
| Access | Closed API | Open weights |
| Alignment training | Heavy RLHF | Lighter |
| Known political lean | Left-libertarian | Less pronounced |
| Represents | Practitioner default for synthetic data | Open-source ecosystem |
| Bias literature coverage | Extensively studied (Santurkar 2023, Röttger 2024) | Studied, less extensively |

This contrast is the minimum viable design to detect generator-specific bias transfer. If both models had identical biases, we couldn't distinguish "bias transfer" from "synthetic data artifact."

**Why GPT-4o-mini specifically (vs. GPT-5.x)?** `gpt-4o-mini` is the model practitioners actually use today for synthetic-data generation pipelines. Studying the model the field uses (rather than the newest available) makes the bias-transfer finding directly actionable. GPT-5.x family models are reserved as a stretch comparator to test whether newer alignment regimes shift the bias profile.

**Why not 3 or 4 generators?** Adding Claude or Grok would push the project past the one-month time budget. We commit to depth-of-analysis (audit, calibration, leakage, error analysis) over breadth-of-models. Future work can extend to a wider model comparison.

---

## Infrastructure

- **Compute:** Drexel CCI cluster (H100 GPUs) for heavy runs; Google Colab Pro+ as backup
- **APIs:** OpenAI (GPT-4o-mini generation, Batch endpoint), HuggingFace (RoBERTa, Llama-3)
- **Libraries:** transformers, peft (LoRA), bitsandbytes, scikit-learn, pandas, sentence-transformers (leakage check)
- **Storage:** Google Drive for checkpoint backup

---

## Timeline (5 weeks) — UPDATED

| Week | Tasks |
|---|---|
| 1 | Data download, generation pipeline setup, pilot runs, HuggingFace Llama-3 access approval |
| 2 | Full synthetic data generation (both GPT-4o-mini and Llama-3); synthetic data audit (lean scoring + leakage check); RoBERTa baseline fine-tuning |
| 3 | Llama-3-8B LoRA fine-tuning; evaluation across all 8 classifiers; bias shift metrics computation |
| 4 | Error analysis (RoBERTa vs Llama-3 disagreement, calibration, qualitative review); midway presentation; paper draft |
| 5 | Final analysis, leakage-filtered ablation, paper writing, final presentation |

---

## Stretch Goal (only if ahead of schedule by end of Week 2)

Add a 3rd generator → 5th training condition → 2 additional fine-tuning runs (or 6 with seeds). Two candidates, in priority order:

1. **GPT-5.4-mini** — tests whether newer OpenAI alignment shifts the bias profile vs. `gpt-4o-mini`. Cost: ~$3.65 batch run. Strong "alignment generation drift" framing.
2. **Claude Haiku 4.5** — adds a third major commercial alignment regime. Cost: ~$5–10.

**Decision criteria:** Only proceed if (a) all Week 1 + Week 2 deliverables are complete by end of Week 2, (b) bandwidth available, and (c) budget allows. Otherwise, mention in Future Work.

**NOT adding:** Grok (intentional bias is a different research question), Mistral (no compelling distinct alignment paradigm), Gemini (less political-bias literature).

---

## Expected Outcomes

The project produces a publishable finding regardless of direction:
- **If bias transfers:** Direct warning to practitioners using LLM-generated data for political NLP.
- **If bias does not transfer:** Null result indicating synthetic data pipelines are more robust than feared.

Both outcomes are valuable and publishable at EMNLP.

---

## Target Venues

**Primary:** EMNLP 2026 Main Conference
- Track: Computational Social Science and Cultural Analytics
- Alternate track: Special Theme on "New Missions for NLP Research"
- ARR submission deadline: May 25, 2026

**Fallback:** EMNLP Findings, ACL 2026, or relevant workshops

---

## Key References

- Feng et al. (2023). *From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models.* **ACL 2023 Best Paper.**
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv.
- Li et al. (2021). *P-Stance: A Large Dataset for Stance Detection in Political Domain.* Findings of ACL.
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv.
- Mohammad et al. (2016). *SemEval-2016 Task 6: Detecting Stance in Tweets.* (Optional secondary dataset, not in current scope)
- Röttger et al. (2024). *Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in LLMs.* ACL 2024.
- Santurkar et al. (2023). *Whose Opinions Do Language Models Reflect?* ICML 2023.
- Wagner et al. (2025). *The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions.* **ICLR 2025 Spotlight.**

---

## Professor Feedback (Round 1) — Addressed

- ✅ Bias shift defined precisely along three axes (class distribution, calibration, directional bias score)
- ✅ Synthetic data audit added (Step 1.5) to characterize political lean of generated tweets
- ✅ Error analysis section added comparing RoBERTa and Llama-3 behavior under bias
- ✅ Limitations section added covering P-Stance label assumption, dataset generalization, generator selection, temporal validity, prompt sensitivity
- ✅ Data leakage explicitly addressed with overlap measurement and optional leakage-filtered ablation
- ✅ Neutral prompt assumption clarified as deliberate (not flaw) — leakage of bias under neutral prompts is part of the hypothesis

---

## Constraints and Design Decisions

### What this project is NOT
- NOT about improving stance detection accuracy
- NOT about mechanistic interpretability
- NOT overlapping with Prof. Shadi Rezapour's lab work (Reddit self-stigma, BioMedBERT)
- NOT requiring any human annotation
- NOT testing on synthetic data
- NOT using GPT-4o full (using cheaper `gpt-4o-mini` variant; bias-transfer signal does not depend on model size)
- NOT including Grok (intentional bias is a different research question)

### Key design choices
- Binary classification (Favor/Against) — P-Stance does not have a Neutral class
- Real tweets as the universal test set — ensures clean comparison
- Two generators with different political profiles — enables directional testing
- Two classifier architectures — tests whether bias transfer depends on model size
- Mixed condition — provides actionable guidance for practitioners
- Synthetic data audit before training — characterizes input distribution
- Leakage-controlled ablation — disentangles memorization from bias transfer
- 3 random seeds per run — enables statistical significance reporting
- Llama-3-Instruct for generation, Llama-3-base for classification — different variants for different roles

---

## Status

- [x] Literature review completed
- [x] Research question finalized
- [x] Proposal drafted
- [x] Dataset class balance verified (P-Stance is binary, not ternary)
- [x] Project proposal submitted to professor
- [x] Professor feedback received and incorporated
- [x] Generator models finalized (GPT-4o-mini + Llama-3-8B-Instruct)
- [x] Scope locked at Version A (2 generators)
- [ ] HuggingFace Llama-3 access requested
- [x] OpenAI API access verified
- [x] Synthetic data generation pipeline built (Batch API)
- [ ] Synthetic data audit pipeline built (lean scoring + leakage check)
- [ ] Baseline RoBERTa fine-tuning
- [ ] Llama-3 LoRA fine-tuning
- [ ] Evaluation and bias shift analysis (3-axis definition)
- [ ] Error analysis (RoBERTa vs Llama-3 comparison)
- [ ] Leakage-filtered ablation (optional)
- [ ] Paper drafted

---

*Last updated: April 2026 — Revision 4 (switched primary OpenAI generator from GPT-5.4-mini to GPT-4o-mini; switched to Batch API; GPT-5.x family moved to stretch goal)*
