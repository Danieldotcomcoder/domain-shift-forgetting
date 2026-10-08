# Study 2 protocol: does TaperNorm increase forgetting where its premise is present?

A pre-registered second-domain test of H1, with an independent web→Python replication

| | |
|---|---|
| Version | 1.0, draft for registration |
| Date | 8 October 2026 |
| Author | Daniel Shdeed |
| Repository | github.com/Danieldotcomcoder/domain-shift-forgetting, branch `study2` |
| Registration | to be filled in at registration (registry, DOI, timestamp) |
| Code freeze | `study2/registration-manifest.json` (SHA-256 of this protocol and of every Study 2 code file) |
| Predecessor | Study 1, the H1 pilot (protocol v3; result STOP, small observed effect). Paper in `paper/` |

**Status at registration.** No Study 2 job has run on any GPU or in any Kaggle session. The domain that Study 2 will use is
not yet known. No activation of any candidate domain has been measured. No outcome of Study 2 exists.
Section 12.2 lists everything that was known or looked at before registration.

[Drafting note for the author: state here whether and how AI assistance was used, as in the pilot paper.]

---

## 1. Background and rationale

**The pilot (Study 1).**
- **Models and conditions.** Small GPT-style models (6 layers, about 17.7M parameters) were trained with RMSNorm
  ("RMS") or with internal TaperNorm ("Taper-minus"). Taper-minus is calibrated over the first 763 updates and faded
  out by a cosine gate that reaches zero at update 6,104. Both conditions keep the final RMSNorm.
- **Training.** Each model was trained on web text for 9,156 updates. From this switch state it continued for 6,104
  updates on either more web text or Python code.
- **Endpoint.** The primary endpoint compared the excess forgetting of held-out web text:

  D = [L(T, python) − L(T, web)] − [L(R, python) − L(R, web)],

  where L is the token-weighted web cross-entropy (nats/token) after 6,104 continuation updates.
- **Result.** Three paired seeds (101–103) gave mean D = −0.0223 nats/token (−0.0200, −0.0207, −0.0262). The
  pre-specified category was **STOP — SMALL OBSERVED EFFECT**: no evidence that TaperNorm increases forgetting
  after a web→Python switch.

**Why a second study.** H1 rests on a mechanism. Training on a new domain changes the scale of the activations
entering the internal normalizers. RMSNorm absorbs such a change, while a tapered TaperNorm, with a frozen scale c
and a gain γ̃, cannot. The pilot's exploratory "Finding A" (paper §5.3) showed that this premise was only weakly
present for Python:
- At the switch, code and web activations entering the 12 internal normalizers differed in scale by a mean absolute
  natural-log ratio of only 0.097 (RMS) and 0.123 (Taper-minus).
- The code-specific part of the scale change during training was about 0.042 (about 4%) in both architectures.
- The largest scale change, a shrinkage of the RMS model by a factor of about 0.81, also happened when training
  simply continued on web text.

A null result for a domain where the premise barely holds says little about H1. The pilot was also narrow in scope:
one domain pair and three seeds.

**What Study 2 does.**
1. **Domain selection.** It tests H1's direction on a new domain X. X is chosen by a rule fixed here: the candidate
   whose activations at the pilot's switch states differ most in scale from web text.
2. **Replication.** It replicates the pilot's web→Python comparison on three fresh, independent seeds (104–106).

Study 2 is a separate study. It does not add seeds to the pilot's analysis, and it changes none of the pilot's
endpoints or decisions. The pilot's protocol forbids outcome-driven extensions, and Study 2 is not one: H2 is a new
question with its own pre-registered rules.

## 2. Questions and hypotheses

**H2 (primary).** After a switch from web text to the selected domain X, does Taper-minus show more persistent
held-out web forgetting than RMS, relative to continuing web training?
- Estimand: D_X at s = 6,104 (Section 8.1).
- Seeds: 101–106.
- Direction (as in H1): D_X > 0 means Taper-minus forgets more.

**R2 (secondary, confirmatory replication of the pilot).** On the fresh seeds 104–106, does the pilot's web→Python
answer replicate? R2 uses the pilot's own estimand, guardrails and decision hierarchy for three seeds.

**M (pre-specified manipulation check, descriptive).** Was the scale premise in fact stronger for X than for
Python? M uses Finding A's three measures (Section 8.6) for X and for Python in the same models.

**Secondary and descriptive analyses (pre-specified).**
- The trajectory D_X(s), F, G and Q.
- Matched-adaptation forgetting.
- Per-document tails, token-class contributions and the rare-token (R) contribution.
- The transient rule.
- The 95% t-intervals and the equivalence reading (Section 8.5).
- A document bootstrap.
- H2 on the fresh seeds alone.
- A pooled six-seed web→Python estimate.
- Per-phase gradient clipping.

All of these are computed by the frozen analysis code. Any other analysis is exploratory and will be labelled as
such.

## 3. Design

**Model, optimizer and conditions.** These are identical to the pilot (Section 6), and the code is copied from the
pilot's runner.

**Seeds and runs.** "Prefix" means the 9,156-update web prefix. Each branch has 6,104 updates.

| Seeds | Prefix | Web branch | Python branch | X branch |
|---|---|---|---|---|
| 101, 102, 103 | reused: the pilot's switch states | reused: the pilot's records | (pilot; used only by the descriptive pooled estimate) | **new** |
| 104, 105, 106 | **new** | **new** | **new** | **new** |

- **Conditions and placement.** Every run is done for RMS and for Taper-minus: RMS on GPU 0 and Taper-minus on GPU 1,
  as independent workers.
- **Pairing.** Within a seed, both conditions start from the same copied initial weights and see the same data order
  (the pilot's `make_state`).
- **Branches.** All branches of a seed and condition restore the identical complete switch state: weights, AdamW
  moments, gradient scaler and RNG state. They continue the pilot's global learning-rate clock from update 9,157 to
  update 15,260.
- **Job order.** Per condition, jobs run in this order: X branch of 101, 102, 103; then 104, 105, 106, each as
  prefix → web → Python → X.

**Token budget.**
- Per condition: 3 × 6,104 + 3 × (9,156 + 3 × 6,104) = 100,716 updates of 16,384 supervised tokens.
- Total over both conditions: **3,300,261,888 supervised tokens**.

**Reuse of pilot assets.** These assets are pinned by hash (Appendix B):
- the pilot's six switch states;
- the pilot's evaluation records for seeds 101–103 (prefix, web and Python branches);
- the pilot's prepared web and Python arrays and its orders for seeds 101–103.

The Study 2 runner verifies every pin before it uses an asset (Section 10). The X branches of seeds 101–103 are
compared with the pilot's own web-branch records, so the comparison requires identical evaluation.
- The evaluation, diagnostics and analysis code is the pilot's, copied verbatim or minimally generalized. A test
  suite checks every copy against the pilot's runner.
- A lineage check re-evaluates each restored switch state before its branch trains (Section 10).

## 4. Selecting the new domain X

### 4.1 Candidates

All four candidates are web-scale text, not gated, under the ODC-BY 1.0 licence plus the Common Crawl terms of use.
Each is pinned to a dataset revision.

| ID | Source (Hugging Face) @ revision | Upstream files | Kind of shift |
|---|---|---|---|
| `mc4-de` | `allenai/c4` @ `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` (the pilot's own C4 revision) | `multilingual/c4-de.tfrecord-*-of-02048.json.gz` (train), `multilingual/c4-de-validation.tfrecord-*-of-00016.json.gz` | same script, different language |
| `mc4-ru` | same | `c4-ru.*-of-04096` (train), `c4-ru-validation.*-of-00032` | Cyrillic script |
| `mc4-zh` | same | `c4-zh.*-of-01024` (train), `c4-zh-validation.*-of-00002` | Chinese script; extreme token shift |
| `openwebmath` | `open-web-math/open-web-math` @ `fde8ef8de2300f5e778f56261843dab89f230815` | `data/train-*-of-00114-*.parquet` (train only) | LaTeX-heavy mathematical web text |

- The `-Latn` transliteration variants of the mC4 languages are excluded by the exact file patterns.
- The candidate order (`mc4-de`, `mc4-ru`, `mc4-zh`, `openwebmath`) is fixed and breaks ties.
- No other programming language is a candidate: the pilot showed that the premise was weak for code.

### 4.2 Probe sample of each candidate

Each probe sample has 256 windows of 513 tokens at stride 512: 131,073 tokens and 131,072 positions, the size of
Finding A's probes. It is built as follows:
- **Document order.** Documents are taken from the candidate's train-side files. The files are ordered by
  SHA-256(`"20260911:" + path`), the pilot's shard order, and rows are read in file order.
- **Eligibility.** A row is eligible if its text is non-empty after stripping whitespace. For OpenWebMath it must also
  be assigned to the training split by the split rule of Section 5.1. So the probe never reads a development or
  reserved-test document.
- **Tokenization.** Each document is GPT-2-tokenized with tiktoken `gpt2` `encode_ordinary`, and one EOS (50,256)
  is appended.
- **Packing.** Documents are concatenated until 256 × 512 + 1 tokens, cutting the last document.

The sample's tokens, document IDs and hashes are written to the probe's output.

### 4.3 Measurement

The probe runs on the six pilot switch states (seeds 101–103 × RMS/Taper-minus) on a Tesla T4 GPU under FP16
autocast with evaluation microbatch 16. It uses the pilot's own diagnostics code unchanged: forward passes only, no
gradient and no update.
- **Energy.** For each of the 12 internal normalizer sites (attention and MLP input norms of the 6 blocks), it
  records E = the mean over all probe positions of ‖h‖², where h is the normalizer's input.
- **Probes.** E is measured on each candidate's sample, on the pilot's web probe (the last 256 windows of the web
  training array) and on the pilot's Python probe (the last 256 windows of the Python training array).

### 4.4 Selection statistic and rule

S(c) = mean over 3 seeds × 2 conditions × 12 sites of |½ ln(E_c / E_web)|

This is exactly Finding A's "code vs web at the switch" measure, averaged over both conditions. **X is the candidate
with the largest S(c).** An exact tie (|ΔS| < 10⁻¹²) goes to the earlier candidate in the fixed order.

The same statistic on the pilot's Python probe, S(python), is the reference. From the pilot's recorded energies it
is 0.1099 (0.0971 RMS, 0.1228 Taper-minus). The probe recomputes it on the GPU.

### 4.5 Validity gate of the probe

Before selecting, the probe must reproduce the pilot's recorded switch-time energies for web and Python at every
site, seed and condition, within a relative difference of 10⁻³. These are the pilot's `prefix@9156` diagnostics.
- If it does not, the probe writes **no selection**, Study 2 stops, and the cause is investigated.
- Any fix needs a documented amendment before the probe is rerun.
- The probe also verifies the SHA-256 of every switch state it loads.

### 4.6 If no candidate exceeds Python's gap

The rule is outcome-blind, so Study 2 proceeds with X whatever S(X) is.
- If S(X) ≤ S(python), the report of H2 must state that the selection did not find a domain with a stronger
  switch-time premise than Python.
- H2's category is then read together with the manipulation check (Section 9.4).
- Study 2 is not cancelled or redesigned in that case. A second domain pair still tests the breadth of H1's
  direction, and the in-training premise (Section 8.6), which a forward probe cannot see, may still differ.

### 4.7 Reporting

The probe reports the following for every candidate:
- S(c);
- S(c) per condition and per model;
- S(c) on each half of the sample (windows 1–128 and 129–256), as a descriptive stability check;
- the ranking.

It also reports the reproduction differences, the hashes of its inputs, and its code and environment. The output is
`selection.json` and `selection.txt` from the Kaggle notebook `s2-domain-probe`.

### 4.8 Known limitation

S is a proxy: a switch-time gap of mean scales. Finding A's key quantity was the scale change during training, which
only training reveals. Section 8.6 therefore measures that change after training as the manipulation check.

## 5. Data for X

The preparation mirrors the pilot's (pilot protocol page 02), with the additions marked **new**. Raw text is
streamed at the pinned revision for preparation only and is never stored. Only GPT-2 tokens and per-document
metadata are kept: source file, row, content SHA-256, URL and split.

### 5.1 Splits

- **mC4 (`c4-official`).** Official train files feed training. Official validation documents go to development or
  reserved test by the pilot's C4 rule: SHA-256 of the document's content hash, modulo 2 (0 → dev, 1 → test).
- **OpenWebMath (`url-group`, new; it has no official held-out split).**
  - Each document's group is SHA-256(`"url:" + URL`), or SHA-256(`"content:" + content hash`) if the URL is empty.
    Pages from the same URL therefore fall in the same split.
  - The pilot's Python rule is then applied to the group: SHA-256 modulo 100, with 0–89 train, 90–94 dev and 95–99
    test.

### 5.2 Tokenization and packing

These are identical to the pilot:
- **Tokenizer.** GPT-2 BPE (tiktoken `gpt2`). Its 50,257-token id-to-bytes map must hash to the pilot's
  `a5623714…dcfdc3d`, otherwise preparation refuses to run. One EOS is appended per document.
- **Order and windows.** Documents are ordered by SHA-256(`"20260911:" + document ID`). Windows have 513 tokens at
  stride 512.
- **Owners.** A per-token owner array maps every token, including EOS, to its document.
- **Splits.** Training splits keep whole windows plus the initial context token. Development and test splits have
  exact token counts. No array crosses a split boundary.
- **Classes.** The token class table (W/A/P and control class "X" in the pilot's table) and the rare flag R (fewer
  than 100 occurrences in the seed's web prefix) are the pilot's, unchanged.
- **Document IDs.** A document's ID is SHA-256(`revision:shard:row`).

### 5.3 Pool expansion

- **Order.** Each upstream is read in the deterministic order of Section 4.2 (shards by SHA-256 order, rows in file
  order; empty texts dropped).
- **Targets.** Reading continues until every split it feeds holds 1.15 × its quota in raw tokens, counting tokens
  plus EOS before deduplication. The stopping check runs after each document.
- **Shortfall.** If deduplication then leaves a split short, its target rises by 10% of its quota and reading
  continues where it stopped. At most 12 expansions are allowed.
- **Determinism.** The pool is a deterministic prefix of the row sequence.

### 5.4 Deduplication

- **Exact duplicates.** Content SHA-256 is compared across the whole X pool and against every document of the
  pilot's online corpus (web and Python, train and development).
- **Near duplicates.** Settings are the pilot's: GPT-2 token 5-gram shingles, MinHash with 128 permutations and seed
  20260911, LSH with 32 bands × 4 rows, then an exact Jaccard check ≥ 0.85 on every candidate pair. Near-duplicates
  are removed within X and against the pilot's web and Python development documents.
- **Precedence.** (new) The pilot's frozen documents always win: any X document that duplicates them, exactly or
  (for development documents) nearly, is removed whatever its split. Within X, test wins over dev, which wins over
  train; ties go to the lower document ID.
- **Audit.** A fixed random audit runs, as in the pilot:
  - up to 10,000 random retained X cross-split pairs;
  - 2,000 random (X document, frozen development document) pairs (new).

  Any pair at or above 0.85 stops preparation.
- **Scope (pre-specified).** Near-duplicates between X and the pilot's training arrays are not searched. X training
  can then only overlap the web *training* data, which leaves the web development set used for D untouched.
  Overlap of X development text with web training text could only make X slightly easier for both conditions
  equally. For mC4 non-English text such overlap is essentially impossible. C4's English filter also removed pages
  containing curly braces, which excludes most LaTeX-heavy pages.

### 5.5 Quotas

| Split | Tokens materialized | Supervised labels |
|---|---|---|
| X train | 110,000,129 (214,844 windows) | 110,000,128 |
| X dev | 2,097,153 | 2,097,152 (quick dev: the first 262,144) |
| X reserved test | 8,388,609 | 8,388,608 |

The X branch needs 6,104 × 32 = 195,328 unique windows. Training documents are never repeated to meet a quota. A
deficit stops preparation (Section 5.9).

### 5.6 Training orders

- **Web and Python orders, seeds 104–106.** These use the pilot's generator, unchanged: NumPy `default_rng(seed)`,
  a permutation of the web windows, then of the Python windows (the first 488,320 and 195,328), plus the R flag from
  the seed's prefix windows.
  - Before writing them, preparation must reproduce the pilot's own order files for seeds 101, 102 and 103 bit for
    bit, otherwise it stops.
  - NumPy 2.5.3 reproduced the seed-101 permutations before registration.
- **X orders, seeds 101–106 (new).** `default_rng([20261008, seed]).permutation(214,844)[:195,328]`, an independent
  stream without replacement.

### 5.7 Reserved test

The X reserved test is materialized and hashed but sealed:
- it is written to a separate folder and recorded only by hash in the training manifest;
- it is excluded from the private Kaggle dataset;
- it is never attached to a training or evaluation notebook.

All Study 2 measurements use development data.

### 5.8 Acceptance checks

Before the arrays are packaged, preparation checks:
- the hashes of every file;
- token/owner alignment and document attribution;
- EOS placement;
- whole training windows;
- exact development and test counts;
- the absence of exact duplicates across splits and against the pilot's corpus;
- the order hashes, uniqueness and ranges;
- the tokenizer identity;
- the selection receipt (the probe's `selection.json`, hashed into the X manifest).

The runner re-verifies every hash at the start of each session.

### 5.9 Pre-specified fallback

If the selected domain cannot meet its quotas within the bounded expansion, it is replaced by the **next candidate
in the probe's ranking** (rank 1). This is recorded as a deviation; no outcome exists at that point. The runner
accepts only rank 0 or rank 1, and verifies that the arrays' domain is that rank's candidate.

## 6. Training (unchanged from the pilot)

- **Architecture.** GPT-style decoder: 6 layers, width 256, 4 heads of 64, GELU MLP width 1,024, context 512,
  learned positions, tied GPT-2 embeddings (50,257), no biases, no dropout.
- **Conditions.** RMS (pre-norm RMSNorm at the 12 internal sites) and Taper-minus (internal TaperNorm, no auxiliary
  loss: calibration to update 763, cosine gate to zero at 6,104). The final RMSNorm is kept in both.
- **Optimizer.** AdamW: learning rate peak 6×10⁻⁴, warmup to update 305, cosine decay to 6×10⁻⁵ at update 15,260 on
  one global clock; betas (0.9, 0.95); eps 10⁻⁸. Matrix weight decay is 0.1 and gains have none. The global gradient
  norm is clipped at 1.0.
- **Batch.** 32 sequences × 512 labels per update, as microbatch 8 × accumulation 4.
- **Precision.** FP16 autocast with GradScaler (initial scale 128) on Tesla T4. Master weights, moments, norm and EMA
  reductions and the loss are FP32.
- **Carried amendments.** The pilot's amendments carry over unchanged: FP16/T4, independent workers, microbatch 8,
  and a self-resuming single notebook.
- **Snapshots.** Weights are kept only for switch states and branch ends. Every full-development point keeps
  per-document and per-class sufficient statistics.

## 7. Evaluation schedule and diagnostics

The schedule is the pilot's (Appendix A): full development at 13 prefix points and 25 continuation points; quick
development at 9,156 and at 80 continuation points.

| Stage | Development sets evaluated | Diagnostic probes (forward only) |
|---|---|---|
| Prefix (104–106) | web, Python | web, Python, X at update 9,156 |
| Web branch (104–106) | web, Python | web, Python, X at s = 0, 10, 100, 1,525, 6,104 |
| Python branch (104–106) | web, Python | same |
| X branch (101–106) | **web, X** | same |

- **Development sets.** Every development set has 4,096 windows (2,097,152 labels). Quick development is its first
  512 windows.
- **Diagnostic probes.** Each probe uses the last 256 training windows of its domain. For web and Python these are
  the pilot's own probes.
- **Use.** Diagnostics are descriptive. They feed the lineage check (Section 10) and the manipulation check
  (Section 8.6), never the decision categories.

## 8. Estimands and analysis

All quantities are computed in-process by the frozen analysis code in `kaggle_s2/s2_run.py`, at the end of every
session. Let L(m, b, s) be the full-development **web** cross-entropy (nats/token) of condition m (T = Taper-minus,
R = RMS) after s updates of branch b (web, or the shifted domain X).

### 8.1 Primary estimand

D_X(s) = [L(T, X, s) − L(T, web, s)] − [L(R, X, s) − L(R, web, s)], with the primary endpoint D_X = D_X(6,104).

The decomposition is the pilot's:
- actual forgetting: F(m, b) = L(m, b, 6,104) − L(m, prefix);
- web-control gap change: G = [L(T, prefix) − L(R, prefix)] − [L(T, web) − L(R, web)];
- direct differential forgetting: Q = D_X − G.

Also reported are the trajectory D_X(s) at all 25 full points, D_X at the persistence points 1,525, 3,050 and
6,104, and quick-development D_X(s) at all 80 quick points.

For seeds 101–103, L(·, web, ·) and L(·, prefix) are the pilot's records; L(·, X, ·) is new.

### 8.2 Guardrails

These are the pilot's guardrails, applied to X:
- **Comparability.** The absolute relative prefix web-CE gap |L(T, prefix) − L(R, prefix)| / L(R, prefix) must be
  ≤ 2%.
- **Adaptation.** X adaptation, the drop in X development non-whitespace CE (classes A, P and the control class)
  from the X branch at s = 0 to s = 6,104, must be ≥ 0.05 nats in both conditions.

The X branch at s = 0 is the switch state itself. For Python (R2) the baseline is the pilot's prefix@9,156 record,
exactly as in the pilot.

### 8.3 Matched adaptation (secondary)

This is the pilot's rule, applied to X:
- **Targets.** For each seed, the targets are RMS's quick X non-W CE at s = 1,525, 3,050 and 6,104.
- **Matching.** Taper-minus is matched at the first chronological downward crossing between adjacent quick points
  no more than 100 updates apart (an exact match uses the observed point). There is no extrapolation and no match if
  a target was already surpassed at s = 0.
- **Interpolation.** Taper-minus's quick web CE is interpolated in X-CE coordinates.
- **Difference.** The matched differential forgetting is reported relative to each model's s = 0 quick web CE. The
  decision uses the furthest target reached by all six seeds.

### 8.4 Tails and token classes (secondary)

These follow the pilot at the endpoint:
- per-document D_X, with an exact reconstruction check;
- the unweighted median;
- the document-count 1%-trimmed token-weighted mean;
- the signed top-1% contribution and its share of net and of positive mass;
- the outlier-concentration flag;
- W/A/P/control-class contributions, which must sum to D_X;
- the rare-token (R) mean difference and its contribution, an overlapping flag.

### 8.5 Uncertainty

- **95% intervals (descriptive).** Seed-level means and sample SDs are reported with 95% t-intervals:
  - multiplier 2.571 (t₀.₉₇₅,₅) for six seeds;
  - multiplier 4.303 (t₀.₉₇₅,₂) for three seeds, as in the pilot.
- **Equivalence reading (secondary inference, new).** The 90% t-interval is mean ± 2.015·SD/√6 for H2 (t₀.₉₅,₅),
  and ± 2.920·SD/√3 for R2 (t₀.₉₅,₂). If it lies entirely inside (−0.015, +0.015) nats/token, the report states
  "equivalent within ±0.015". This is two one-sided t-tests at α = 0.05.
  - The bound is the pilot's "small effect" investment threshold. It was not derived from a utility analysis, and
    the reading is conditional on the corpus, recipe and model scale.
  - It is read only when the decision category is neither INVALID nor COMPARABILITY / ADAPTATION LIMITED.
- **Document bootstrap (descriptive).** Development documents are resampled with replacement, with the same
  resample for all arms and seeds (B = 10,000, seed 20261011). It measures evaluation-sample noise with the trained
  models held fixed, never seed-level uncertainty.

**Sensitivity.** Six seeds are fixed by the compute budget, not chosen for power. Assume the pilot's between-seed SD
of D (0.0034) holds. Then the six-seed 95% half-width is about 0.0036, and the 90% half-width about 0.0028:
- an effect of +0.03 would be distinguished from 0 with large margin;
- a true effect near 0 would very likely satisfy the equivalence reading.

If the SD were three times larger (0.010), the 90% half-width would be about 0.0082. Equivalence would then require
|mean| < 0.0068, and an effect of +0.03 would still be clearly distinguished. These are planning figures, not
guarantees.

### 8.6 Manipulation check: Finding A's measures (descriptive, pre-specified)

The measures use the diagnostic probes. They are natural-log ratios of per-site scales; absolute values are taken
per site and seed, then averaged over 12 sites × seeds, per condition.
- **Switch gap.** |½ ln(E_dom / E_web)| at the switch state. For X it comes from the X branch's s = 0 diagnostics,
  for Python from prefix@9,156.
- **Branch change of the web-probe scale.** ½ ln(E_web(end of branch) / E_web(prefix@9,156)) for the web branch and
  for the shifted branch, absolute and signed, with counts of site–seed pairs that shrank.
- **Domain-specific change.** The shifted-branch change minus the web-branch change, per site and seed, absolute.

The check is computed for X (seeds 101–106) and Python (101–106: the pilot's records for 101–103). Pre-specified
readings:
- **"Premise stronger for X at the switch"** if X's mean absolute switch gap exceeds Python's in both conditions.
- **"Premise stronger for X in training"** if X's mean absolute domain-specific change exceeds Python's in both
  conditions.

These readings never change a decision category. They govern how H2 is interpreted (Section 9.4). The probe's
selection statistic (Section 4.4) is reported alongside; it uses a different X sample (the probe sample, not the
last 256 training windows).

### 8.7 R2: the replication

R2 is the pilot's analysis, unchanged, on seeds 104–106 with Study 2's own web and Python branches:
- D, D(3,050), Q, F and G;
- the prefix gap and the Python non-W adaptation from prefix@9,156;
- matching on quick Python non-W CE, the tails, the classes and R;
- the 3-seed 95% interval (4.303) and the equivalence reading (2.920).

### 8.8 Pooled web→Python estimate (descriptive only)

This is the mean, SD and 95% interval (2.571) of D over seeds 101–106. It is labelled as including the pilot's
already-published seeds 101–103, and it is never a decision input.

As an integrity check, the pilot seeds' D recomputed from the pilot's records must equal the pilot's published
values within 10⁻¹².

### 8.9 Further descriptive outputs

- H2 on the fresh seeds 104–106 alone (mean, SD and 3-seed intervals). Seeds 101–103 share their switch states with
  the selection probe, so this shows whether the result depends on them.
- Per-phase gradient-clipping fractions: the prefix calibration, gate-decay and gate-zero phases, and each branch.
- The prefix web-CE slopes over the last five prefix points, for seeds 104–106.

## 9. Decision rules

### 9.1 H2 (seeds 101–106, all thresholds as in the pilot)

The rules are applied in order after all twelve Study 2 runs are complete.

| # | Category | Condition | H2 reading |
|---|---|---|---|
| 1 | Invalid or incomplete | Missing seed or record; numerical failure; or a correctness or data gate failed: a pin, the data hashes, a lineage check or a record integrity check | No answer |
| 2 | Comparability / adaptation limited | Any seed has prefix gap > 2%, or X adaptation < 0.05 nats in either condition | No answer |
| 3 | Opposite direction | Mean D_X ≤ −0.03 | **No** (Taper-minus forgets less) |
| 4 | Proceed (H2 supported at screening level) | Mean D_X ≥ 0.03; D_X > 0 in all six seeds; mean D_X(3,050) ≥ 0.015; mean Q > ½ mean D_X; mean matched differential forgetting > 0 at the furthest common target | **Yes** |
| 5 | Transient only | Mean D_X < 0.015, and the mean quick-development D_X(s) ≥ 0.05 at some s ≤ 1,000 | **No** (not persistent) |
| 6 | Stop: small observed effect | Mean D_X < 0.015 | **No** |
| 7 | Inconclusive | Anything else | Undecided; no automatic extra seeds |

### 9.2 R2 (seeds 104–106)

R2 uses exactly the pilot's hierarchy (the same table, with three seeds, D for the Python branch and Python
adaptation). It is applied when the fresh seeds' web and Python branches are complete.
- If R2 is STOP or OPPOSITE (or TRANSIENT), the pilot's NO replicates.
- If R2 is PROCEED, it does not replicate.

### 9.3 Equivalence statements

The equivalence reading (Section 8.5) is the only basis on which Study 2 may say an effect is "no larger than
0.015 nats/token". Category 6 alone remains "not proof of equivalence", as in the pilot.

### 9.4 Interpretation matrix (pre-registered)

| H2 category | Manipulation check | Permitted reading |
|---|---|---|
| Proceed | premise stronger for X | TaperNorm shows persistent excess forgetting where the scale premise is present (screening level, this setting) |
| Proceed | premise not stronger | Excess forgetting on X, not explained by the measured scale premise |
| Stop, opposite or transient | premise stronger for X | Evidence against the scale-mismatch account of H1 in this setting. With the equivalence reading, "no excess larger than 0.015" |
| Stop, opposite or transient | premise not stronger | A second domain pair without excess forgetting. Uninformative about the mechanism, because the premise was again weak |
| Inconclusive, limited or invalid | any | No conclusion about H2 |

"Premise stronger" refers to the in-training reading, reported together with the switch-time reading.

No result of Study 2 is a population claim beyond this model scale, recipe and corpus sample.

## 10. Validity gates and failure policy

1. **Pilot pins.** Before use, the runner verifies against Appendix B:
   - the pilot's configuration hash and experiment ID;
   - each switch state's SHA-256 (file and receipts);
   - each evaluation record file's canonical digest.
2. **Data.** Every array, owner file, document manifest and order file is hash-verified at each session. The X
   arrays must have been deduplicated against this exact pilot corpus, with the same tokenizer and class table.
   The domain must be the probe's recorded selection (rank 0), or the pre-specified fallback (rank 1).
3. **Lineage (new).** At s = 0 of every branch, before any update, the restored switch state is re-evaluated and
   compared with the records of that same state: the pilot's prefix@9,156 records for seeds 101–103, and the run's
   own for 104–106.
   - Full and quick web-development CE must agree within 10⁻³ nats/token.
   - The per-site web and Python diagnostic energies must agree within a relative 10⁻³.
   - On failure, the worker stops before training, the failure is recorded, and the decisions become INVALID. An
     investigation and an amendment are required.

   The pilot's own records show bit-identical re-evaluations of identical weights, so any difference measures
   software or environment drift.
4. **Numerical failure.** A non-finite loss, gradient or weight after six FP16 retries is a numerical failure. It
   invalidates a clean H2 reading. Seeds are never replaced and thresholds never changed.
5. **Infrastructure interruptions.** A session that runs out of time saves state and exits. The next session
   resumes from the latest complete checkpoint (rotated every 15 minutes, and at every stage boundary).
6. **Integrity checks.** Every evaluation record must have the right update clock, label population, class and
   document reconstruction, and parent switch state. A failure is reported and invalidates the decisions; it is never
   silently dropped.
7. **No outcome-dependent changes.** There are no interim analyses with consequences, no extra seeds, no changed
   endpoints or thresholds, and no selective reporting. Every planned run and every recorded failure is reported.

## 11. Execution plan

All compute runs on Kaggle, and nothing runs on the author's computer.

1. **Probe.** Notebook `s2-domain-probe`: GPU T4, internet on, about 0.3 h. It streams the four probe samples and
   measures the six switch states.
2. **Prepare.** Notebook `s2-prepare-domain`: CPU only, internet on, about 1–3 h. It prepares X, runs the acceptance
   checks and writes the orders.
3. **Private dataset.** `danny00/study2-online-inputs` holds `s2online/`, `s2orders/` and `s2upstream/`, never the
   reserved test.
4. **Smoke test.** Separate notebook `s2-single-notebook-smoke`, GPU T4 × 2, about 0.75 h. It produces no outcome
   data:
   - (A) the mini protocol end to end, on a mini "pilot" made by the pilot's own runner, with interruptions and
     resumes;
   - (B) the lineage check of all six real switch states (forward passes only);
   - (C) real-protocol throughput and resume on seed 104's web prefix only.
5. **Main run.** Notebook `s2-single-notebook-run`, GPU T4 × 2, self-resuming. First a CPU bootstrap that authorizes
   one fresh start, then GPU sessions.

**Environment.** All GPU notebooks are pinned to the Kaggle Docker image of the pilot's run notebook
(PyTorch 2.11.0+cu128, CUDA 12.8), if Kaggle still offers it. Otherwise the then-current image is used, and the
deviation is recorded. Each session records Python, PyTorch, CUDA, the GPUs and the runner's SHA-256.

**Compute budget.** This comes from the pilot's measured timings: about 61 min per branch and 65 min per prefix per
GPU, evaluations included.
- Per GPU: 3 × 61 + 3 × (65 + 3 × 61) ≈ 927 min ≈ 15.5 h.
- Sessions: two, the first capped at 11.25 h.
- Total including the probe and smoke test: about 16.5–17 h of the 30 h weekly Kaggle T4 × 2 quota.

There is no GPU-hour cap beyond Kaggle's free quota. A quota pause is an infrastructure interruption.

**Stopping rules.** Training stops only:
- when the experiment is complete;
- for a numerical, lineage or data failure;
- when Kaggle resources are exhausted (resume later);
- by explicit decision of the author, which will be reported.

## 12. Registration, transparency and amendments

### 12.1 What is registered

- This protocol.
- `study2/registration-manifest.json`: the SHA-256 (LF line endings, as run on Kaggle) of this protocol and of every
  file in `kaggle_s2/`; the pilot runner's hash; the candidate pins; the Kaggle notebook and dataset names; and the
  pinned Docker image.
- The git commit containing them.

Every Study 2 session logs its runner's SHA-256, and the status script compares it with the registered hash.
Optionally, the Study 2 notebooks are made public at registration time for an independent server-side timestamp.

### 12.2 Known before registration

- **Pilot.** Everything about the pilot: its results, records and paper, including Finding A and the pilot seeds'
  prefix and branch evaluations.
- **Source checks.** To pin the candidates, the Hugging Face API was queried for dataset revisions, file listings,
  file sizes and licence cards. OpenWebMath's row order was spot-checked through the datasets-server API: the hosts,
  dates and lengths of 60 rows, to confirm that rows are not grouped by site. No candidate text was tokenized, and
  no activation of any candidate was measured.
- **Order generator.** NumPy 2.5.3 was checked locally to reproduce the pilot's seed-101 web and Python orders.
- **Code tests.** The Study 2 code was tested only on synthetic data and on the pilot's committed records, on CPU.

### 12.3 Not known before registration

- the selected domain;
- any candidate's scale statistic;
- any X-branch result;
- any result for seeds 104–106.

### 12.4 Amendments and deviations

- **Before the first primary-run update.** A change is allowed only through a dated amendment appended to this file,
  with the reason and the affected step, before that step runs, and with the registration updated.
- **After the first primary-run update.** No scientific change is allowed. Infrastructure fixes that do not change
  the science are documented as deviations.
- **Pre-specified responses.**
  - A pinned revision becomes unavailable: stop, then an amendment with the new revision.
  - The probe's reproduction gate fails: stop, investigate, then an amendment.
  - The selected domain falls short of its quotas: rank-1 fallback.
  - The pilot's Docker image is unavailable: the current image, recorded.

## 13. Limitations

- **Scale.** One small model (17.7M parameters, embedding-dominated), one optimizer recipe, FP16 on T4. Effects may
  not transfer to larger models.
- **Seeds.** Six seeds estimate seed variance only roughly. Seeds 101–103 reuse the pilot's prefixes, which also
  served the domain selection.
- **Selection proxy.** The selection proxy measures mean activation scale at the switch. Normalizers also act on
  per-token variation, which S does not capture.
- **Same-seed comparison.** For seeds 101–103 the X branch is compared with web-branch records produced in an
  earlier session. The lineage check and environment pinning bound, but cannot exclude, all differences between
  sessions.
- **Data.** One prepared sample per domain: uncertainty is conditional on it. Deduplication is approximate, and
  near-duplicates with the pilot's training arrays are not searched (Section 5.4).
- **Thresholds.** The decision thresholds (0.015 and 0.03) were set as investment thresholds in the pilot, not
  derived from a power or utility analysis.
- **Development data only.** The reserved test sets of the pilot and of X stay sealed for any later confirmatory
  study.

## 14. Outputs and reporting commitments

**Published outputs.**
- `selection.json`;
- the X manifest, audit and acceptance record;
- `config.json` with its hash;
- `decision-report.txt` and `decision-report.json`, which contain both decisions, all seed values, intervals, the
  equivalence readings, the manipulation check, lineage results, clipping, missing records and failures;
- every evaluation record (`events.jsonl`) and training log (`train.jsonl`);
- the session records.

Switch states and final weights (several GB) stay in the Kaggle notebook output.

**Reporting commitments.**
- Every decision is reported with its pre-registered wording (Section 9.4).
- Results are reported whatever their direction, including INVALID, LIMITED or INCONCLUSIVE outcomes.
- The study will be reported as a separate, pre-registered study alongside the pilot, which remains unchanged.

---

## Appendix A. Evaluation schedule (identical to the pilot)

- **Prefix full development.** Updates 763, 3,052, 6,104, 6,409, 6,714, 7,019, 7,324, 7,629, 7,934, 8,239, 8,544,
  8,849, 9,156. Quick development at 9,156. Diagnostics at 9,156.
- **Continuation quick development (80 points).** s ∈ {0, 1, 2, 5, 10, 20}; every 50 from 50 to 1,000; every 100 from
  1,100 to 6,100; plus 1,525, 3,050 and 6,104.
- **Continuation full development (25 points).** s ∈ {0, 1, 10, 100}; every 305 from 305 to 6,100; plus 1,525,
  3,050 and 6,104.
- **Diagnostics.** s ∈ {0, 10, 100, 1,525, 6,104}. Persistence points and matching targets: 1,525, 3,050 and 6,104.

## Appendix B. Pinned pilot assets

Source: the Kaggle notebook `danny00/h1-single-notebook-run` and the dataset `danny00/stage1-online-inputes`. Copies
of the records are in `reports/h1-kaggle/`.

| Asset | SHA-256 |
|---|---|
| Pilot configuration (`config_sha256`) | `c18441799eee2b6a78b257e2e22c6ac32ddb7c30cd81d6eb57e488b2b173ce10` |
| Pilot experiment ID | `c18441799eee2b6a-6ac0f2cb` |
| Pilot runner `kaggle_h1/h1_run.py` | `66769e7de30ec36f13a5acfa275d2691cb30251a08b9354a5c5063bd813cb83b` |
| Pilot online manifest | `290886d93760f5a66f1e55b3c4730a02dfddf1b8153ba12ed78fba6374ab87d7` |
| Pilot orders manifest | `bd7abf44121a7402acc1d9603d3d29f20a1074a97cd202e0e2f964406bcc9bac` |
| GPT-2 id-to-bytes map | `a5623714bcf19049eb0fd78df19b6daa1e61ac435b86a5deae767768e7fcdc3d` |

The six switch states (complete prefix states at update 9,156) and the canonical digests of their evaluation records
are listed below. A canonical digest is the SHA-256 of each record re-serialized as sorted-key compact JSON, one per
line, so it does not depend on line endings.

| Run | Switch state `switch.pt` | Evaluation records `events.jsonl` (canonical digest) |
|---|---|---|
| S101 RMS | `6e890478d28826c4ba020e674717b87ffbbd3dd363d478876b96ef944f0a292d` | `2d8aced7560ca17f28fb6b70c19d7fa519920dbe99234dab570fcdb98cd4e02f` |
| S101 Taper-minus | `223da6d23881954c96928e36d9d5e74270ae16ab296ce0dbffb7a45078f727c2` | `9a013705584eef64ce6ef16c7fe3b0e55037b794491ab6c56234466e12fa216b` |
| S102 RMS | `0f0f603bf4546d1955f0e75fe3c4dc6af1514dc8bb01c0166f1830b739a82b12` | `53060b9a3f5e95ebc47a3a9640142e0efaf5b7befef3c8f5b707cf988d455e49` |
| S102 Taper-minus | `78fde02da7e9d922edf6918e86487193931a83331196f4f1bbf58c620db251b2` | `8f782648ef6759b7c26c586cb26a6b0922663adedfaf8e403f0d497e39e43c28` |
| S103 RMS | `e7b5c32566bc6ba137f52492f27475891bceb9c0fbfe85bd7ae43b6c706ea94c` | `65f3caf2ce6016d8f475b19946e317f575e014453ac423432ae5444c840b894c` |
| S103 Taper-minus | `1e457bd6fb889c7a717deac80f3eb9e85a5e007bed7728f587a501c8c80e9f5d` | `d0df7be0c88a3a6ca3a0a8e52e603fe2d73c374f9412c302d375c01386663ea0` |

The same pins are the `PILOT_PINS` constant in `kaggle_s2/s2_run.py`, covered by the registration manifest. A test in
`kaggle_s2/test_s2_run.py` checks them against the committed pilot records.

## Appendix C. Notation

| Symbol | Meaning |
|---|---|
| T, R | The Taper-minus and RMS conditions |
| X | The selected domain |
| Control class | The pilot's token class "X": EOS, control bytes and invalid standalone UTF-8 fragments |
| s | Continuation updates after the switch (global update 9,156 + s) |
| L | Token-weighted web cross-entropy on 2,097,152 held-out development labels (nats/token) |
| Non-W CE | Cross-entropy over labels in classes A, P and control (whitespace-only tokens excluded) |
