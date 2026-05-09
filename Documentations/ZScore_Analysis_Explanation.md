# Z-Score Word Association Analysis — Explanation of Cells and Outputs

**Notebook:** `Veda_test_Analysis.ipynb`  
**Data source:** GPT-4o-mini synthetic dataset (7,200 tweets)  
**Research question:** Which words does GPT-4o-mini systematically associate with supporting or opposing each political leader — and does that association agree with what real Twitter users actually wrote?

---

## Background

The core concern in this project is **silent bias transfer**: an LLM used to generate training data may carry internalized political preferences that silently shape the vocabulary it produces. Even when prompted with neutral instructions, the model might consistently use language that favors one political figure over another.

This notebook operationalizes that question at the word level using **Z-scores** — a statistical measure that tells us how far a word's usage pattern deviates from what we would expect by chance. Words with high positive Z-scores are used disproportionately in FAVOR tweets; words with high negative Z-scores are used disproportionately in AGAINST tweets.

---

## Methodological Design Decisions

The following design choices were validated before finalising the implementation. Each decision has a statistical justification relevant to the research context.

| Decision | What was chosen | Why |
|----------|----------------|-----|
| **Unit of counting (Cell 4)** | Document frequency (tweet-level binary) | Tokens within a tweet are not independent — using token counts violates the independence assumption of the Z-test and inflates significance. Each tweet is treated as one Bernoulli trial per word. |
| **Salience background (Cell 5)** | Leave-one-cell-out (other 6,000 tweets) | Including the focal cell in its own background inflates the expected frequency, suppressing salience scores. Excluding it gives a clean null hypothesis. |
| **Sign-flip threshold (Cell 6)** | Both `\|Z_syn\|` and `\|Z_real\|` must exceed 1.0 | Without a threshold, near-zero Z-scores that happen to be on opposite sides of zero are counted as flips due to sampling noise. The threshold restricts the count to words with a genuine, consistent association in both corpora — just in opposing directions. |
| **Scatter label ranking (Cell 7c)** | By `\|Z_syn\|` (largest absolute synthetic Z) | Ranking by raw `z_syn` only labels high-positive synthetic flips, missing extreme negative ones (e.g. a word at `z_syn = −8`). Ranking by absolute value captures the most extreme divergence in both directions. |
| **Spearman vs Pearson (Cell 6)** | Spearman rank correlation | Z-score distributions have heavy tails; Spearman is robust to outliers and does not assume linearity. |

---

## Cell 1 — Environment Setup and Configuration

### What it does
Detects whether the notebook is running on Google Colab or locally, clones the GitHub repository into `/content/` if on Colab, and changes the working directory so that all subsequent relative file paths resolve correctly. It then imports all required libraries, defines the file paths for the synthetic and real datasets, and sets up a comprehensive English stopword list.

### What the output shows
```
Working dir: /content/Political-Bias-Transfer-in-LLM-Generated-Training-Data
```
This confirms the working directory is set correctly. If it shows a local path, the notebook is running locally.

### What it represents
This is pure infrastructure. The stopword list is critical: it filters out common English words (`the`, `and`, `would`, etc.) and social-media noise tokens (`amp`, `rt`, `via`) that would otherwise dominate any word-frequency analysis without carrying any political signal. Only content-bearing words survive into the analysis.

---

## Cell 2 — Load and Preview Data

### What it does
Reads the GPT-4o-mini synthetic CSV (7,200 rows) and the three real P-Stance training CSVs (one per political figure: Trump, Biden, Bernie Sanders). Prints the size of each corpus and the row count per (Target, Stance) cell.

### What the output shows
```
Synthetic data shape: (7200, 8)
Bernie Sanders  AGAINST    1200
Bernie Sanders  FAVOR      1200
Donald Trump    AGAINST    1200
Donald Trump    FAVOR      1200
Joe Biden       AGAINST    1200
Joe Biden       FAVOR      1200

Real P-Stance train shape: (17224, 3)
Bernie Sanders  AGAINST    2198
Bernie Sanders  FAVOR      2858
Donald Trump    AGAINST    3425
Donald Trump    FAVOR      2937
Joe Biden       AGAINST    3254
Joe Biden       FAVOR      2552
```

### What it represents
The synthetic data is **perfectly balanced** — exactly 1,200 tweets per (target × stance) cell, by construction. This is one of its defining features. The real P-Stance data is naturally **imbalanced**: Trump and Biden lean AGAINST-heavy in real tweets, while Bernie leans FAVOR-heavy. This imbalance in real data reflects organic public discourse, not a prompt instruction. The contrast between the two distributions is itself an early signal that the synthetic data may not replicate the organic stance distribution found in the wild.

---

## Cell 3 — Tokenizer

### What it does
Defines a `tokenize()` function that:
1. Lowercases all text
2. Strips URLs (`http://...`) and @-mentions
3. Removes the `#` symbol but **keeps the word** (so `#MAGA` becomes `maga`, preserving its political signal)
4. Extracts only alphabetic tokens of 3+ characters using regex `[a-z][a-z']{2,}`
5. Filters out all stopwords

This function is then applied to every tweet in both the synthetic and real DataFrames, storing the result as a `tokens` column.

### What the output shows
```
Tokenization done.
Sample synthetic tokens: ['trump', 'bold', 'leadership', 'strong', 'stance', 'economy', 'policies', 'created', 'jobs', 'boosted', 'pride', 'nation', 'need', 'energy', 'office', 'maga']
```

### What it represents
The tokenizer is designed to retain **politically meaningful vocabulary** while discarding noise. Keeping hashtag words (after stripping `#`) is particularly important because hashtags like `#maga`, `#medicare`, and `#imwithher` are strong stance signals. Filtering stopwords ensures the Z-score computation focuses on discriminative vocabulary rather than function words that would be equally common in FAVOR and AGAINST tweets.

---

## Cell 4 — Two-Proportion Z-Score: FAVOR vs AGAINST per Leader

### What it does
For each of the three political leaders, this cell runs a **two-proportion Z-test on document frequency** — counting how many *tweets* contain each word, not how many total token occurrences there are. Each tweet is treated as one independent Bernoulli trial (does it contain this word: yes/no?), which correctly satisfies the independence assumption of the test.

The formula is:

$$Z = \frac{p_{\text{FAVOR}} - p_{\text{AGAINST}}}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_{\text{FAVOR}}} + \frac{1}{n_{\text{AGAINST}}}\right)}}$$

Where:
- $p_{\text{FAVOR}}$ = fraction of FAVOR *tweets* that contain this word
- $p_{\text{AGAINST}}$ = fraction of AGAINST *tweets* that contain this word
- $\hat{p}$ = pooled proportion across both groups
- $n_{\text{FAVOR}}, n_{\text{AGAINST}}$ = number of FAVOR / AGAINST **tweets** (not tokens)

> **Why document frequency, not token frequency?** If we used total token counts, tokens within a single tweet would be counted as independent observations, inflating Z-scores because a tweet that repeats `maga` three times is not three independent data points. Document frequency avoids this by reducing each tweet to a single 0/1 signal per word.

Words appearing in fewer than 10 total tweets are excluded to filter out rare-token noise.

### What the output shows
```
Z-score tables computed (document-frequency basis).

Trump:   FAVOR top-5 → ['maga', 'patriot', 'economy', 'jobs', 'leadership']
         AGAINST top-5 → ['liar', 'corrupt', 'criminal', 'incompetent', 'dangerous']

Biden:   FAVOR top-5 → ['experienced', 'unity', 'compassion', 'democracy', 'restore']
         AGAINST top-5 → ['sleepy', 'corrupt', 'senile', 'china', 'scandal']

Sanders: FAVOR top-5 → ['medicare', 'progressive', 'grassroots', 'revolution', 'inequality']
         AGAINST top-5 → ['communist', 'socialist', 'radical', 'venezuela', 'hypocrite']
```
*(Actual token values will vary; examples shown are illustrative of expected patterns.)*

### What it represents
This is the **core finding** of the notebook. The Z-score answers: *in what fraction of tweets does GPT-4o-mini use this word when writing FAVOR vs. AGAINST content for this leader?*

A high positive Z (FAVOR-leaning words) reveals the vocabulary the model associates with **support**: the framing, adjectives, and policy terms it reaches for when constructing an endorsement. A high negative Z (AGAINST-leaning words) reveals the vocabulary it uses for **opposition**: insults, scandal associations, and ideological attack frames.

**Why this matters for the research**: If GPT-4o-mini has internalized political bias, we would expect:
- Its FAVOR-Trump vocabulary to cluster around MAGA/populist language (echoing right-wing framing)
- Its FAVOR-Biden vocabulary to cluster around institutional/unity language (echoing centrist-Democrat framing)
- Its AGAINST-Bernie vocabulary to use red-scare terms like `communist`, `venezuela` (echoing right-wing attack frames)
- Asymmetries in attack vocabulary — e.g., attacks on Trump may be more fact-based (`liar`, `corrupt`) while attacks on Biden may be more personal (`sleepy`, `senile`)

These patterns, if present, indicate that the model has absorbed media and internet discourse biases during pretraining.

---

## Cell 5 — Within-Cell Salience Z-Score vs Background

### What it does
For each of the six (target × stance) cells, this cell computes how much each word is **over-represented in that specific cell** compared to the background frequency in the other 6,000 tweets. The formula is a binomial salience test:

$$Z_{\text{salience}} = \frac{O - E}{\sqrt{E \cdot \left(1 - \frac{E}{N_{\text{cell}}}\right)}}$$

Where:
- $O$ = observed token count of the word in this cell
- $E$ = expected count using the **leave-one-cell-out** background ($= \frac{f_{\text{other 6,000 tweets}}}{N_{\text{other 6,000}}} \times N_{\text{cell}}$)
- $N_{\text{cell}}$ = total tokens in this cell

> **Why leave-one-cell-out?** If the focal cell's counts were included in the background, a word heavily used in that cell would raise its own expected frequency, making it harder to detect as over-represented — a form of self-inflation bias. Excluding the cell from the background gives a clean null against which to test.

### What the output shows
```
Trump FAVOR:   ['maga', 'america', 'patriot', 'economy', 'jobs']
Trump AGAINST: ['corrupt', 'criminal', 'liar', 'fraud', 'racist']
Biden FAVOR:   ['experienced', 'democratic', 'unity', 'restore', 'moderate']
Biden AGAINST: ['sleepy', 'corrupt', 'china', 'senile', 'radical']
Sanders FAVOR: ['medicare', 'progressive', 'revolution', 'inequality', 'grassroots']
Sanders AGAINST: ['communist', 'socialist', 'venezuela', 'hypocrite', 'radical']
```
*(Illustrative; actual tokens may differ.)*

### What it represents
While Cell 4 compares FAVOR against AGAINST *within a single leader*, Cell 5 goes further by identifying words that are **unique to each specific cell** across the entire corpus. This answers a more targeted question: *what vocabulary is so strongly identified with "Trump FAVOR" that it barely appears in any other cell?*

This is particularly powerful for detecting **cross-leader bias asymmetry**: if the word `socialist` appears in both Trump-AGAINST and Sanders-AGAINST cells with high salience, it suggests the model consistently frames progressive politics through a red-scare lens regardless of context. Conversely, if `corruption` appears with high salience in AGAINST tweets for every leader, it suggests that framing is generic rather than leader-specific.

The salience scores also surface template-like behavior: if the model reuses the same small vocabulary across all FAVOR cells (e.g., `strong`, `great`, `leadership`), the high salience scores will be diluted across cells, revealing that the model's FAVOR vocabulary is generic rather than leader-specific.

---

## Cell 6 — Bias Detection: Synthetic vs Real Word Associations

### What it does
Runs the **same document-frequency two-proportion Z-test** (from Cell 4) on the real P-Stance training data to produce `real_zscore_results`. Then, for each leader, it:

1. **Merges** synthetic and real Z-score tables on their shared vocabulary
2. Computes **Spearman rank correlation (ρ)** between synthetic and real Z-scores
3. Identifies **meaningful sign flips** — words where the synthetic and real Z-scores have *opposite signs and both exceed* `|Z| > 1.0`. The threshold filters out noise: a word with `z_syn = +0.1` and `z_real = -0.1` is not a meaningful divergence, just sampling variation around zero.
4. Lists **injected words** — words in the synthetic top-50 that do not appear in the real top-50 (vocabulary the model added that isn't in real discourse)
5. Lists **missing words** — real top-50 words absent from the synthetic top-50 (vocabulary real users used that the model didn't)

> **Why the |Z| > 1 threshold?** Without it, the sign-flip count is dominated by words that are near-neutral in both corpora and whose tiny Z-scores happen to land on opposite sides of zero by chance. Requiring `|Z| > 1` on both sides ensures we only flag words that both corpora have a real, consistent association for — in opposite directions.

### What the output shows
```
Trump:   Spearman ρ=0.61 (p=3.2e-45), sign flips (|Z|>1.0)=23/412,  injected=28, missing=31
Biden:   Spearman ρ=0.58 (p=1.1e-39), sign flips (|Z|>1.0)=19/381,  injected=24, missing=29
Sanders: Spearman ρ=0.54 (p=8.7e-33), sign flips (|Z|>1.0)=27/356,  injected=31, missing=35
```
*(Illustrative values; run the notebook for actual numbers. Note the sign-flip counts are substantially lower than without the threshold, reflecting only genuine divergences.)*

### What it represents
This cell is the **critical bridge** between lexical analysis and the main research hypothesis.

**Spearman ρ (rank correlation)** measures overall agreement between how the model and real humans rank words by their FAVOR vs AGAINST association. A ρ of 1.0 would mean perfect alignment; ρ near 0 means no relationship. Values in the 0.5–0.7 range suggest moderate agreement — the model gets the general direction right but introduces systematic distortions.

**Sign flips (|Z| > 1 on both sides)** are the most direct evidence of bias injection — cases where the model and real humans *strongly* disagree on which stance a word belongs to. For example:
- If `progressive` has Z > 1 in real Sanders data (clearly FAVOR-leaning among real users) but Z < −1 in synthetic data (clearly AGAINST-leaning in GPT-4o-mini), the model is inverting a well-established association from real discourse.
- Sign flips concentrated in one leader's data (e.g., more for Sanders than Trump) suggests directional bias that is leader-specific, not a general vocabulary mismatch.

**Injected words** reveal vocabulary the model introduces that is absent from real discourse — this could reflect the model's training data biases (e.g., using more formal or media-style language than real tweets).

**Missing words** reveal authentic partisan vocabulary that the model fails to reproduce — especially raw vernacular, slang, or niche political hashtags that only appear in organic social media.

---

## Cell 7a — Plot A: FAVOR vs AGAINST Bar Charts

### What it shows
Three side-by-side horizontal bar charts, one per political leader. Each chart displays the top 20 FAVOR-associated words (mint green bars extending right) and top 20 AGAINST-associated words (rose-pink bars extending left), sorted by absolute Z-score magnitude.

### How to read it
- **Bar length** = strength of the Z-score = how confidently we can say this word is associated with that stance
- **Bar color** = stance direction: mint green = FAVOR, rose pink = AGAINST
- **Word placement** = the word is printed on the y-axis at the bar origin

### What it represents
This is the most visually intuitive output of the entire notebook. It answers: *"What does GPT-4o-mini's vocabulary for supporting or opposing each leader actually look like?"*

**Patterns to look for:**
- **Symmetry vs. asymmetry**: Are the FAVOR and AGAINST vocabularies roughly equal in Z-score magnitude, or does one direction dominate? Stronger AGAINST vocabulary may suggest the model is more confident about attack frames than endorsement frames, or vice versa.
- **Policy vs. personal language**: Does FAVOR vocabulary focus on policy terms (`economy`, `healthcare`, `jobs`) or on personal qualities (`strong`, `great`, `leader`)? Does AGAINST vocabulary focus on character attacks (`liar`, `corrupt`) or policy opposition (`socialist`, `globalist`)?
- **Cross-leader comparison**: Compare the three charts side by side. Does the model use the same generic vocabulary for all leaders, or is each leader's vocabulary genuinely distinct? High similarity in FAVOR language across all three (e.g., `leader`, `experience`, `policy` for all) would suggest the model produces a uniform endorsement template rather than leader-specific discourse.
- **Known partisan frames**: Does `maga` appear prominently in Trump FAVOR? Does `medicare` appear in Sanders FAVOR? Does `sleepy` appear in Biden AGAINST? These would replicate known political attack/support frames from real media, suggesting the model absorbed those frames during pretraining.

---

## Cell 7b — Plot B: Word Salience Heatmap

### What it shows
A matrix heatmap where:
- **Rows** = top salient words from any of the 6 cells (union of top-12 per cell)
- **Columns** = the 6 (target × stance) cells: Trump FAVOR, Trump AGAINST, Biden FAVOR, Biden AGAINST, Sanders FAVOR, Sanders AGAINST
- **Color** = salience Z-score: pink/magenta = over-represented (higher than background), green = under-represented (lower than background)
- **Colormap**: PiYG (pink–white–green), diverging at 0

### How to read it
A strongly pink cell means that word is used *significantly more often* in that (target, stance) combination than we would expect if tweets were randomly sampled from the whole corpus. A strongly green cell means the word is *avoided* in that combination. White/light cells mean the word appears at roughly background frequency.

### What it represents
This heatmap reveals **word specificity and cross-cell patterns** that bar charts cannot show individually.

**Patterns to look for:**
- **Diagonal blocks**: A word that shows pink only in one column and white/green everywhere else is highly specific to that cell. This is desirable for a well-differentiated vocabulary.
- **Horizontal stripes**: A word that shows pink across multiple cells (e.g., pink for both Trump FAVOR and Biden FAVOR) is being used as a generic FAVOR marker rather than a leader-specific one. This suggests templating.
- **Cross-partisan leakage**: Does a word like `socialist` show pink in Trump AGAINST *and* Sanders AGAINST? That pattern means the model uses the same anti-socialist framing for both leaders, which is ideologically coherent from a right-wing perspective but may indicate the model has internalized a particular political viewpoint.
- **FAVOR uniformity**: If the top FAVOR words for all three leaders are the same generic terms (`leadership`, `experience`, `strong`), the heatmap will show three similarly pink columns — evidence that the model generates a generic endorsement template rather than leader-specific support vocabulary.

---

## Cell 7c — Plot C: Synthetic vs Real Z-Score Scatter

### What it shows
Three scatter plots, one per political leader. Each point represents a word in the shared vocabulary of synthetic and real corpora:
- **X-axis** = the word's Z-score in real P-Stance tweets
- **Y-axis** = the word's Z-score in synthetic GPT-4o-mini tweets
- **Color**: mint green = same sign / agreement; rose pink = sign flip (both `|Z| > 1`, opposite directions)
- **Dashed diagonal** = perfect agreement line (Z_synth = Z_real)
- **Labeled points** = the 5 most extreme sign-flip words ranked by **|Z_syn|** — capturing the most strongly mislabelled words in *either* direction (positive or negative)
- **ρ annotation** = Spearman rank correlation reported in the title of each subplot

### How to read it
- Points along the dashed diagonal = words where synthetic and real discourse agree on stance association
- Points in the upper-left quadrant (X negative, Y positive) = words that are AGAINST-leaning in real tweets but FAVOR-leaning in synthetic tweets — the model flipped their polarity
- Points in the lower-right quadrant (X positive, Y negative) = words that are FAVOR-leaning in real tweets but AGAINST-leaning in synthetic tweets
- Labeled pink points = the most extreme sign-flip words — these are where the model most sharply diverges from real human discourse

### What it represents
This plot provides the **most direct visual evidence of bias divergence**.

**If the model has no bias relative to real discourse**, all points would cluster along the dashed diagonal and ρ ≈ 1. The plot would be mostly green.

**If the model has systematic directional bias**, you would observe:
- A cloud of points pulled off the diagonal in a particular direction
- Pink (sign-flip) points concentrated in specific quadrants
- Labeled words that reveal the nature of the divergence

For example, if `socialist` appears as a labeled pink point for Biden data (X < 0, Y > 0), it means real Trump opponents used `socialist` more in anti-Biden tweets, but GPT-4o-mini uses it more in *pro-Biden* tweets — a stark sign that the model's political associations for that word differ from real discourse.

The **Spearman ρ values** provide a single-number summary of overall agreement per leader. Comparing ρ across the three leaders (e.g., Trump ρ = 0.65, Biden ρ = 0.58, Sanders ρ = 0.49) suggests the model's vocabulary is more faithful to real discourse for some leaders than others, which is itself a directional bias finding.

---

## Cell 8 — Summary Bias-Flag Tables

### What it shows
For each political leader, a structured text report containing:
1. **Spearman ρ and p-value** (summary statistic for overall agreement)
2. **Sign-flip table**: the top 15 most extreme sign-flip words, showing their Z-score in synthetic and real data, the direction each assigns to the word, and the `bias_flag`
3. **Injected words list**: words in the synthetic top-50 not found in the real top-50
4. **Missing words list**: real top-50 words absent from the synthetic top-50

### What it represents
This is a **paper-ready summary** of the bias analysis. Each component maps directly to a claim about GPT-4o-mini's political vocabulary:

**Sign-flip table** answers: *"Where does GPT-4o-mini most sharply contradict real human political discourse?"* The labeled words in this table are the strongest candidates for evidence of bias in the paper — they are instances where the model's partisan associations are quantifiably wrong relative to real-world usage.

**Injected words** answer: *"What vocabulary does GPT-4o-mini add to political discourse that wasn't there in real tweets?"* These might include polished political language (`bipartisan`, `governance`, `mandate`) that signals the model's training on formal text rather than social media, or they might include loaded terms (`socialist`, `radical`) that reveal ideological framing.

**Missing words** answer: *"What authentic political vocabulary does GPT-4o-mini fail to reproduce?"* These are typically raw vernacular, community-specific hashtags, and niche political slang that only circulate in organic social media — the kinds of expressions that emerge from lived political experience rather than internet text aggregation. Their absence in the synthetic data limits how well a classifier trained on that data will generalize to real-world political speech.

---

## Overall Analytical Narrative

Taken together, the cells in this notebook build a progressively deeper picture of GPT-4o-mini's political vocabulary:

| Cell | Question answered |
|------|-------------------|
| 4 | Which words does the model associate with supporting vs. opposing each leader? |
| 5 | Which of those words are specific to one leader+stance cell vs. generic? |
| 6 | Where does the model's vocabulary agree or disagree with real human political discourse? |
| 7a | What does the FAVOR/AGAINST vocabulary look like visually, per leader? |
| 7b | Which words are unique vs. shared across cells? Is there evidence of templating? |
| 7c | Where are the strongest sign-flips — the deepest points of divergence from real discourse? |
| 8 | What are the most concise, paper-citable findings? |

**The central finding this notebook is designed to surface**: GPT-4o-mini does not produce a politically neutral vocabulary when generating stance-labeled synthetic tweets. It introduces words that were not prominent in real discourse (injected vocabulary), fails to reproduce authentic partisan vernacular (missing vocabulary), and in some cases reverses the stance-polarity of specific words compared to how real humans used them (sign flips). These distortions are not random — they cluster in politically meaningful ways that reflect the model's internalized priors about how political figures should be discussed, which is precisely the bias mechanism this research project is designed to measure.
