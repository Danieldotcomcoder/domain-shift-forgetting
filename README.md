# Domain-Shift Forgetting: Does TaperNorm Forget More?

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Hardware](https://img.shields.io/badge/run%20on-Kaggle%20T4%20x2-20beff)

Two pre-specified studies ask whether replacing a transformer's internal RMSNorm layers with **TaperNorm**
(normalization that is gradually switched off during training) makes the model forget more of what it learned
when its training data switches domain.

> **H1.** After switching training from web text to Python code, does a model trained with internal
> TaperNorm (*Taper-minus*) lose more held-out web performance than an otherwise identical RMSNorm model,
> relative to simply continuing web training?

- **Study 1** is a small screening pilot of H1 (web → Python, three paired seeds).
- **Study 2** is a separate study. It tests H1's direction on a new domain chosen by a fixed rule (**H2**), and it
  replicates the pilot on three fresh seeds (**R2**). It also measures H1's premise, a domain-induced change in
  activation scale, as a pre-specified manipulation check.

In both studies the protocol, data splits, endpoints and decision rules were fixed before any training run. Both
are screening studies, not confirmatory results.

**Paper:** a preprint describing both studies is in [`paper/main.pdf`](paper/main.pdf) (source and asset
generator in [`paper/`](paper/)).

## Results at a glance

D is the difference-in-differences in held-out web cross-entropy (nats/token) at the endpoint. Positive D means
extra forgetting under TaperNorm.

| Study | Shift | Seeds | Mean D | Descriptive 95% interval | Pre-specified decision |
|---|---|---|---:|---|---|
| 1 | web → Python | 101–103 | −0.0223 | [−0.0308, −0.0139] | **STOP — SMALL OBSERVED EFFECT** |
| 2 (H2) | web → Chinese (mC4 zh) | 101–106 | −0.0325 | [−0.0683, +0.0032] | **OPPOSITE DIRECTION** |
| 2 (R2) | web → Python | 104–106 | +0.0093 | [−0.0084, +0.0271] | **STOP — SMALL OBSERVED EFFECT** (the pilot's NO replicates) |

No study found the excess forgetting H1 predicts, and none showed equivalence within ±0.015 nats/token.

## Study 1 result

> **H1: No.** Taper-minus did not show more persistent web forgetting than RMSNorm after the switch to Python.
> The pre-specified decision is **STOP — SMALL OBSERVED EFFECT**.

The run was complete and valid. All three paired seeds finished, no events were missing, there were no
numerical failures, and every guardrail passed.

| Seed | D (endpoint) | D at 3050 | Q | Matched forgetting (furthest target) | Prefix gap | Code improvement (RMS / Taper) |
|---|---:|---:|---:|---:|---:|---:|
| 101 | −0.0200 | +0.0109 | −0.0194 | −0.0308 | 0.28% | 3.55 / 3.55 nats |
| 102 | −0.0207 | −0.0124 | −0.0199 | −0.0183 | 0.28% | 3.60 / 3.46 nats |
| 103 | −0.0262 | +0.0193 | −0.0220 | −0.0286 | 0.34% | 3.35 / 3.34 nats |
| **Mean** | **−0.0223** | **+0.0059** | **−0.0204** | **−0.0259** | | |

- **Mean D:** −0.0223 nats/token (SD 0.0034). The descriptive t95 interval is [−0.0308, −0.0139].
- **Guardrails passed:** prefix gaps of 0.28–0.34% (limit 2%) and code improvements of about 3.3–3.6 nats
  (minimum 0.05).

**How to read it**

- **Both models forget heavily.** Switching to Python raises held-out web cross-entropy by about +1.63 to
  +1.69 nats/token for both conditions, while continuing on web lowers it by about 0.13. The difference
  between conditions (D) is about 1.3% of that forgetting.
- **D points the opposite way to H1.** It is negative in all three seeds, meaning Taper-minus forgot slightly
  *less*. The difference sits in the Python branch itself: the web-control gap change G is about 0, so Q ≈ D.
- **This is not a finding that TaperNorm protects against forgetting.** The effect is smaller than the
  pre-specified "opposite direction" threshold (mean D ≤ −0.03). Its sign was also unstable earlier in
  continuation: D at 1525 was +0.037, +0.022 and −0.029 across the seeds, and D at 3050 was mixed.
- **No early transient.** The largest early mean quick-dev D (update ≤ 1000) was +0.015, far below the 0.05
  transient threshold.
- **The effect is broad, not driven by outliers.** Median per-document D is about −0.02 to −0.03 and the top 1%
  of documents do not dominate. Alphanumeric tokens contribute negatively in every seed and most in two of
  three; rare tokens contribute at most 3% of D.
- **What the rules allow.** STOP means no convincing persistent excess forgetting within this pilot. It is not
  evidence of equivalence. Per protocol, no extra seeds, changed endpoints or follow-up study start
  automatically. Study 2 is a new question with its own protocol, not an extension of the pilot.
- **Hindsight from Study 2.** The three seeds agreed closely, but that understated the seed-to-seed variation:
  three fresh seeds gave positive web → Python contrasts, all outside this interval (see below).

**Exploratory analyses** (chosen after the results; descriptive only, see the paper's §5.3–5.4)

- D(s) had two seed-consistent negative phases separated by a noisy middle: a brief early dip (s = 100: mean
  −0.0185, seed SD 0.0021) and the final low-learning-rate phase (s ≥ 5,185: mean −0.020, negative in all 15
  seed–point values). In between it stayed near zero.
- The premise of H1 was weak in these models: the code-specific change in activation scale at the internal
  normalizer inputs was about 4% per site in both architectures, and the larger scale change in the RMS model
  (a shrinkage by a factor of about 0.81) happened to a similar extent when training simply continued on web
  text.
- A document bootstrap gives a mean-D interval of [−0.0235, −0.0211], conditional on the trained models;
  seed-to-seed variation is about 3.5× larger than this evaluation noise (comparing like with like).

**Run record**

- **One Kaggle session:** 9.36 h on 2x Tesla T4, PyTorch 2.11 / CUDA 12.8, with zero worker restarts.
- **Configuration:** sha256 `c18441…` is recorded in [`config.json`](reports/h1-kaggle/h1state/config.json).
- **Evidence in this repository** (`reports/h1-kaggle/`):
  [`decision-report.txt`](reports/h1-kaggle/h1state/decision-report.txt) and
  [`decision-report.json`](reports/h1-kaggle/h1state/decision-report.json), the session record, the notebook
  log, and for every run the evaluation records (`events.jsonl`: per-document and per-class sufficient
  statistics and diagnostic probes), the per-update training logs (`train.jsonl`) and completion receipts.
- **Withheld:** the model checkpoints (switch states and branch-final weights, about 2 GB) stay in the output of
  the private Kaggle notebook `danny00/h1-single-notebook-run`. The prepared token arrays are not
  redistributed; the preparation code regenerates them from the pinned dataset revisions.
- **Pre-specification:** the decision thresholds are bound into the configuration hash the run logged at
  12:19:23 UTC, before any outcome existed, and the runner script has the same SHA-256 in every commit. The
  first public commit came after seed 101's endpoint values had appeared in the live log; the paper (§3.6)
  discloses this timeline in full.

## Study 2 result

The protocol is [`study2/PROTOCOL.md`](study2/PROTOCOL.md); the execution log is
[`study2/RUNLOG.md`](study2/RUNLOG.md).

**Domain selection.** Four candidates were fixed in advance: German, Russian and Chinese mC4, and OpenWebMath.
The rule picks the candidate whose activations at the pilot's switch states differ most in scale from web text.
**Chinese (mC4 zh)** was selected, with S = 0.0934, ahead of Russian (0.0893), German (0.0514) and OpenWebMath
(0.0119). No candidate exceeded Python's S of 0.1099, a case the protocol anticipated (§4.6): Study 2 proceeds,
and the H2 report must say so.

> **H2 (web → Chinese, seeds 101–106): OPPOSITE DIRECTION.** Mean D_X = −0.0325 nats/token (SD 0.0341;
> 95% interval [−0.0683, +0.0032]; 90% interval [−0.0606, −0.0045]; equivalence within ±0.015 not shown).

| Seed | D_X | D_X at 3050 | Q | Matched (6104) | Prefix gap | Chinese improvement (RMS / Taper) |
|---|---:|---:|---:|---:|---:|---:|
| 101 (pilot switch state) | −0.0886 | −0.0260 | −0.0880 | −0.1137 | 0.28% | 2.30 / 2.33 nats |
| 102 (pilot switch state) | +0.0065 | −0.0176 | +0.0074 | −0.0466 | 0.28% | 2.31 / 2.33 nats |
| 103 (pilot switch state) | −0.0351 | −0.0739 | −0.0309 | −0.0854 | 0.34% | 2.23 / 2.27 nats |
| 104 | −0.0513 | +0.0071 | −0.0487 | +0.0460 | 0.26% | 2.21 / 2.24 nats |
| 105 | −0.0179 | −0.0070 | −0.0153 | −0.0130 | 0.23% | 2.18 / 2.19 nats |
| 106 | −0.0089 | −0.0856 | −0.0082 | −0.0143 | 0.30% | 2.28 / 2.31 nats |
| **Mean** | **−0.0325** | **−0.0338** | **−0.0306** | **−0.0378** | | |

- **A narrow call.** The mean is just past the −0.03 threshold and the 95% interval includes 0. On the fresh
  seeds alone (104–106) the mean is −0.0260 (95% [−0.0816, +0.0296]), which does not reach −0.03. Leaving
  out seed 101 or seed 104 (post hoc) would make the category STOP. Every leave-one-out mean stays below
  +0.015, though, so the pre-specified reading below would be the same: only the label is fragile.
- **Large seed variation.** The SD of D_X is about 10 times the pilot's SD of D. A single seed's D_X(s) moves
  by about 0.04 between adjacent evaluations (post hoc).
- **Near-catastrophic shift.** The Chinese branch raised web cross-entropy from about 4.58 to about 6.99
  nats/token in both architectures (forgetting 2.34–2.55 nats/token, against 1.63–1.69 for Python). Under
  GPT-2's tokenizer, 85% of the Chinese labels are byte fragments.
- **Not evidence of protection.** We read the category as no evidence for H1 on Chinese, not as evidence that
  TaperNorm protects against forgetting.

> **R2 (web → Python, fresh seeds 104–106): STOP — SMALL OBSERVED EFFECT.** Mean D = +0.0093 (SD 0.0071;
> 95% [−0.0084, +0.0271]; 90% [−0.0027, +0.0214]). The pilot's decision replicates.

- **The sign does not replicate.** D was positive in every fresh seed (+0.0160, +0.0102, +0.0018) and negative in
  every pilot seed, and no fresh value lies inside the pilot's interval.
- **Pooled six-seed estimate** (descriptive; includes the pilot's published seeds): mean D −0.0065,
  95% [−0.0254, +0.0125].

**Post hoc one-sided reading** (not pre-specified). Is an excess of +0.015 ruled out? The one-sided 95% upper
bounds are:

- **Chinese:** −0.0045, so yes. Equivalence failed only on the lower side.
- **Fresh Python seeds alone:** +0.0214, so no.
- **Pooled Python:** +0.0084, so yes.

**Manipulation check (pre-specified, descriptive).** All values are mean absolute natural-log ratios of
activation scales, RMS / Taper-minus.

- **At the switch, the premise was not stronger for Chinese.** The domain-vs-web gap was 0.085 / 0.122 for
  Chinese and 0.093 / 0.136 for Python.
- **During training, the premise was about 4× stronger for Chinese.** The domain-specific change of the
  web-probe scale was 0.165 / 0.183 for Chinese and 0.039 / 0.042 for Python.
- **The two architectures moved in opposite directions on Chinese.** The web-probe scale shrank by a factor
  of 0.72 in RMS and grew by a factor of 1.21 in Taper-minus.
- **Caveat.** The in-training measure is taken after treatment, in models that forgot different amounts. A
  larger change may partly be a symptom of heavier forgetting, not an independent premise.

**Pre-specified reading** (Protocol §9.4): opposite direction, with the in-training premise stronger for X, is
*evidence against the scale-mismatch account of H1 in this setting*. Equivalence was not shown, so the protocol
does not allow the stronger statement "no excess larger than 0.015".

**Run record**

- **Compute.** A domain probe, a CPU data preparation and a smoke test, then the main run: two Kaggle sessions
  of 11.12 h and 4.06 h on 2x Tesla T4, with PyTorch 2.11.0 in the pilot's pinned Docker image. There were no
  restarts, failures, missing records or FP16 retries.
- **Lineage.** All 24 lineage checks were bit-exact: every restored switch state re-evaluated exactly to its
  records, including the pilot's six in a new session.
- **Evidence** (`reports/s2-kaggle/`): the probe's `selection.json`, the preparation manifest, audit and
  acceptance records, the smoke-test results, the decision reports, the session records, worker logs, and every
  run's `events.jsonl`, `train.jsonl` and receipts.
- **Independent recomputation.** [`paper/make_assets.py`](paper/make_assets.py) recomputes every contrast,
  interval and manipulation-check value from these records and asserts agreement with the decision report.
- **Withheld.** Checkpoints stay in the Kaggle notebook output. The preparation notebook stays private, because
  its output holds the sealed reserved test split.

**Pre-specification (Protocol Amendment 1).** Study 2 ran before a public registry entry was filed.
- **The manifest.** [`study2/registration-manifest.json`](study2/registration-manifest.json) holds the SHA-256 of
  the protocol and every Study 2 code file.
- **Submitted for timestamping before any job.** The manifest's hash went to four OpenTimestamps calendars at
  21:55 UTC on 8 October 2026, one minute before the first Study 2 job.
- **Embedded in every notebook version.** Every Study 2 notebook embeds the manifest byte for byte. The versions
  were created 21:56–22:59 UTC, so Kaggle records server-side times for them.
- **The Bitcoin attestation came later.** The proof ([`.ots`](study2/registration-manifest.json.ots)) is anchored
  in Bitcoin block 970,569 (01:59 UTC on 9 October). That is after the Chinese branches of seeds 101–103 had
  finished inside the running session, so the earlier bound rests on Kaggle's records.
- **Inspecting the proof.** Run `python study2/timestamp.py info study2/registration-manifest.json.ots`.
  [`study2/timestamp-evidence.json`](study2/timestamp-evidence.json) collects the facts above.

Read Study 2 as "specified in advance and hash-timestamped before any run, with public registration deferred until after execution".

## Status

| Stage | State |
|---|---|
| Study 1: protocol v3, data specification, analysis plan | Frozen |
| Study 1: corpus preparation (C4 English + The Stack dedup Python, GPT-2 tokens) | Done, hash-verified |
| Study 1: implementation + equivalence tests | Done |
| Study 1: Kaggle smoke test (T4 x2) | Passed 2026-10-03 |
| Study 1: primary run (3 seeds x 2 conditions x 3 phases) | Complete 2026-10-03 (12:18–21:40 UTC) |
| **Study 1: decision on H1** | **No: STOP — SMALL OBSERVED EFFECT** |
| Study 2: protocol, Amendment 1, code | Frozen 2026-10-08 21:54 UTC; manifest hash timestamped |
| Study 2: domain probe, preparation, smoke test | Done 2026-10-08 (mC4 zh selected; preparation accepted; smoke passed) |
| Study 2: main run (12 condition–seed runs) | Complete 2026-10-10 01:40 UTC |
| **Study 2: decisions** | **H2: OPPOSITE DIRECTION. R2: STOP — SMALL OBSERVED EFFECT (replicates)** |
| Study 2: public registry entry | To be filed after execution (Amendment 1) |

To run Study 2 yourself, follow [`kaggle_s2/README.md`](kaggle_s2/README.md). The sections below describe the
shared design and Study 1's runner; Study 2 reuses them unchanged.

## Experimental design

### Model and conditions

- A GPT-style decoder: 6 layers, width 256, 4 heads, MLP width 1024 (GELU), context 512, learned positions,
  tied GPT-2 embeddings (vocabulary 50,257), no biases, no dropout. About 17.7M parameters.
- **RMS:** standard pre-norm RMSNorm at all 12 internal sites.
- **Taper-minus:** internal TaperNorm (no auxiliary loss), following
  [TaperNorm, §3 and Appendix C](https://arxiv.org/html/2602.10408v1). It is calibrated during the first 763
  updates, faded out with a cosine gate until update 6,104, and fully off afterwards.
- Both conditions keep the final RMSNorm.
- Seeds 101, 102 and 103 are **paired**: both conditions start from the same copied initial weights and see the
  same data order.

### Training timeline (one condition, one seed)

```text
             web prefix (9,156 updates)              ┌── web branch:    +6,104 updates of web text
update 0 ──────────────────────────────────────── 9,156
             305: LR warmup ends                     └── Python branch: +6,104 updates of Python code
             763: Taper calibration frozen                (both branches restore the identical switch state)
           6,104: Taper gate reaches zero
```

- **Batch:** every update has 32 sequences x 512 labels = 16,384 tokens.
- **Optimizer:** AdamW at lr 6e-4 with cosine decay to 6e-5 on one global clock.
- **Weight decay and clipping:** matrix decay 0.1, no decay on gain vectors, gradient clipping at 1.0.
- **Primary allocation:** 2.10B supervised tokens.

### Primary endpoint

With $L$ the token-weighted web cross-entropy (nats/token) on 2,097,152 held-out development labels, measured
after 6,104 continuation updates:

$$
D = \big[L(\text{Taper},\text{python}) - L(\text{Taper},\text{web})\big] - \big[L(\text{RMS},\text{python}) - L(\text{RMS},\text{web})\big]
$$

**Positive D means Taper-minus suffers more excess web forgetting under the Python shift.** The analysis also
reports actual forgetting F, the web-control gap change G, the direct differential forgetting Q = D − G,
matched-adaptation forgetting, per-document tails and per-token-class contributions.

### Pre-specified decision rules

The rules are applied in this order after all three paired seeds complete:

| # | Category | Condition | H1 reading |
|---|---|---|---|
| 1 | Invalid or incomplete | Missing seed, numerical failure, or correctness/data failure | No answer |
| 2 | Comparability / adaptation limited | Prefix web-CE gap > 2%, or a code branch improves < 0.05 nats | No answer |
| 3 | Opposite direction | Mean D ≤ −0.03 | **No** |
| 4 | Proceed to design the next study | Mean D ≥ 0.03, D > 0 for every seed, mean D(3050) ≥ 0.015, mean Q > ½ mean D, matched forgetting > 0 | **Yes** (pilot level) |
| 5 | Transient only | Mean D < 0.015, but an early mean quick-dev D ≥ 0.05 (update ≤ 1000) | **No** (not persistent) |
| 6 | Stop: small observed effect | Mean D < 0.015 | **No** (not proof of equivalence) |
| 7 | Inconclusive | Anything else | Undecided |

With three seeds the reported 95% t-interval (mean D ± 4.303·s/√3) is descriptive only.

## Running the experiment

The whole experiment is one file, [`kaggle_h1/h1_run.py`](kaggle_h1/h1_run.py). It runs as a Kaggle notebook on
two T4 GPUs: RMS on GPU 0 and Taper-minus on GPU 1, each processing seeds 101 → 102 → 103.

- **Checkpoints persist automatically.** All state goes to the notebook's own output folder (`h1state/`). The
  notebook lists its own output as an input, so each new version resumes from where the last one stopped.
- **No hand-off between sessions.** A session that runs out of time saves and exits cleanly; you just start
  the notebook again.

### Prerequisites

- A Kaggle account with an API token ([kaggle.com/settings/api](https://www.kaggle.com/settings/api)).
  Store it in a local `.env` as `KAGGLE_API_TOKEN=...`, which is git-ignored.
- Python 3.11+ with `pip install -e ".[model,test]" kaggle`.
- The prepared corpus as a **private** Kaggle dataset containing `online/` and `orders/`. See
  [Data preparation](#data-preparation).

### Launch, monitor, continue

```bash
# 1. Point kaggle_h1/build_notebooks.py at your Kaggle username and dataset (USER, DATASET).

# 2. First push only: a CPU bootstrap (~1 min, no GPU quota) that authorizes one fresh start.
python kaggle_h1/build_notebooks.py run --phase bootstrap
kaggle kernels push -p kaggle_h1/kernel-run

# 3. Every later push: the GPU run (T4 x2), which resumes itself from the previous output.
python kaggle_h1/build_notebooks.py run --phase gpu
kaggle kernels push -p kaggle_h1/kernel-run

# 4. Check progress and fetch the decision report. --continue starts the next session if needed.
python kaggle_h1/h1_status.py [--continue]
```

An optional smoke test runs the full pipeline on a tiny schedule plus 25 minutes of real training. It uses a
separate notebook (`python kaggle_h1/build_notebooks.py smoke`, then push `kaggle_h1/kernel-smoke`), so it
never touches the experiment's state.

### Compute

These figures come from the Kaggle smoke run with both workers running at once:

| Measurement | Value |
|---|---|
| Training throughput, RMS | ~45k tokens/s per T4 |
| Training throughput, Taper-minus | ~42–43k tokens/s per T4 |
| One full-development evaluation (both domains) | ~37 s |
| Projected total | ~9–10 h per GPU, typically one 11¼-hour session |
| Actual primary run (2026-10-03, UTC) | 9.36 h, one session, no restarts |

Kaggle's weekly GPU quota applies, and T4 x2 may be billed above 1x elapsed time.

### Outputs

```text
h1state/
  config.json               frozen configuration, data hashes, amendments (sha256-bound)
  decision-report.txt|json  decision category, H1 reading, per-seed D, D(3050), Q, guardrails, tails
  sessions.jsonl            one line per Kaggle session (runtime, GPUs, exit codes)
  runs/S{seed}-{condition}/
    events.jsonl            every scheduled evaluation with per-document/per-class sufficient statistics
    train.jsonl             per-update loss, gradient norm, clipping, FP16 scale
    switch.pt               complete prefix state shared by both branches
    web-final.pt, python-final.pt, completion.json
```

## Data preparation

The corpus is prepared once on CPU, before any GPU time. Raw text is never committed to this repository.

- **Web:** [allenai/c4](https://huggingface.co/datasets/allenai/c4), `en`, train split for training; the
  official validation split is hashed 50/50 into development and reserved test.
- **Code:** [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup), `data/python`.
  Access is gated and needs an approved Hugging Face token. Files are grouped by repository alias before
  splitting, by `sha256(group) mod 100`: 0–89 train, 90–94 dev, 95–99 test.
- **Deduplication:** exact content hashes across all splits, plus a MinHash/LSH near-duplicate audit
  (5-gram shingles, 128 permutations, 32x4 bands, Jaccard ≥ 0.85). Held-out documents always win.
- **Quotas:** 260M web + 110M Python training tokens; 2,097,152 development labels per domain; 8,388,608
  reserved-test labels per domain. The reserved test set is sealed and is never attached to a training run.

```bash
python -m domain_shift_forgetting.pilot sources --output prep/upstream
python -m domain_shift_forgetting.pilot_data inspect --output prep/access          # needs HF_TOKEN
python -m domain_shift_forgetting.pilot_data prepare --access prep/access/access.json \
       --output prep/corpus --cache prep/cache                                    # resumable
python -m domain_shift_forgetting.pilot orders --data-dir prep/corpus/online --output prep/orders
```

Upload `prep/corpus/online/` and `prep/orders/` (plus `prep/upstream/`) as one private Kaggle dataset.
Keep `prep/corpus/reserved-test/` elsewhere. A Kaggle CPU-notebook version of these steps is in
[`notebooks/kaggle-stage1-prepare.ipynb`](notebooks/kaggle-stage1-prepare.ipynb).

## Testing

```bash
pytest                              # synthetic unit tests for src/ (protocol, data, norms, model, analysis)
pytest kaggle_h1/test_h1_run.py     # h1_run.py vs. the tested src/ implementation
```

`kaggle_h1/test_h1_run.py` checks that the runner matches the reference package:

- identical schedules, learning-rate and gate clocks;
- identical paired initialization;
- identical training updates and TaperNorm forward passes in every gate state;
- vectorized evaluation that reproduces the reference evaluator on the real development data;
- identical decision categories across 3,000 random evidence sets;
- checkpoint resume within GPU nondeterminism.

The CUDA and real-data tests skip when no GPU or prepared corpus (`data/kaggle-online/`) is available. The
mini end-to-end test is GPU-heavy; prefer the Kaggle smoke notebook.

## Protocol amendments

All amendments were recorded before any primary-run update. None changes the model, data, token budgets,
seeds, endpoints or decision thresholds.

| Date | Amendment |
|---|---|
| 2026-09-28 | Tesla T4: FP16 autocast + GradScaler replaces BF16. FP32 master weights, moments, norm/EMA reductions and loss. Results are never pooled with a BF16 run. |
| 2026-09-30 | Free Kaggle GPU time across sessions replaces the original 24 paid GPU-hour cap. |
| 2026-10-01 | RMS and Taper-minus run as independent workers on two T4s, with no shared state. |
| 2026-10-03 | Microbatch 8 x accumulation 4 (the protocol default; same effective batch). Single self-resuming notebook. Weight snapshots are kept only for the switch state and branch endpoints, for storage; every full-dev point keeps its sufficient statistics. |

## Repository layout

```text
kaggle_h1/                 Study 1 runner: h1_run.py, notebook builder, status script, equivalence tests
kaggle_s2/                 Study 2: runner, domain probe, domain preparation, notebook builder, status script,
                           tests, and the runbook (README.md)
study2/                    Study 2 protocol (with Amendment 1), registration manifest, OpenTimestamps proof,
                           timestamp evidence, execution log, hand-off notes and the OSF filing guide
src/domain_shift_forgetting/
  models/                  Transformer, RMSNorm, TaperNorm
  training/                Effective update, checkpoints, budget arithmetic
  evaluation/              Token-weighted CE sufficient statistics, diagnostics helpers
  analysis/                Contrasts (D, F, G, Q), matching, tails, class contributions, decision rules
  data/                    Repository grouping, splits, dedup, packing, token classes
  pilot_data.py, pilot.py  Corpus preparation and order generation CLIs
tests/                     Synthetic unit tests for src/
configs/                   Protocol values (stage1.v3.json) and earlier Kaggle configurations
docs/                      Implementation notes and earlier runbooks
notebooks/, scripts/       Earlier Kaggle workflows (superseded by kaggle_h1/ for execution)
Domain-Shift Forgetting · Stage 1 — H1 Research Pi/   Protocol export: the scientific source of truth
reports/h1-kaggle/         Study 1: decision report, frozen config, session record, run log, and per-run
                           evaluation records, training logs and completion receipts
reports/s2-kaggle/         Study 2: probe selection, preparation records, smoke-test results, decision
                           reports, session records, worker logs, and every run's records
paper/                     Preprint source (both studies), compiled PDF, and the script that generates its numbers
manifests/, registry/      Manifest templates and run/task registries
data/, artifacts/          Local data and checkpoints (contents git-ignored)
```

### Protocol documents

Study 2's protocol is [`study2/PROTOCOL.md`](study2/PROTOCOL.md). For Study 1, the original protocol pages
remain the scientific source of truth:

- [Project overview](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)
- [01 · Protocol v3, H1 only](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/01%20%C2%B7%20Protocol%20v3%20%E2%80%94%20H1%20only%203d88a2d10e44812ea734fdeff5891429.md)
- [02 · Data, tokenization and leakage controls](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/02%20%C2%B7%20Data,%20tokenization%20and%20leakage%20controls%203d88a2d10e44815391dffb1e1533faa8.md)
- [03 · Analysis plan and decision rules](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/03%20%C2%B7%20Analysis%20plan%20and%20decision%20rules%203d88a2d10e4481af9a3deb8e29c432c7.md)
- [04 · Execution, GPU budget and reproducibility](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/04%20%C2%B7%20Execution,%20GPU%20budget%20and%20reproducibility%203d88a2d10e44811ebd95cc0afce540c9.md)
- [05 · Decision sheet, limitations and revision record](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/05%20%C2%B7%20Decision%20sheet,%20limitations%20and%20revision%20reco%203d88a2d10e4481298fd9f24f8a9b7309.md)

## Scope and limitations

- **Narrow questions.** Study 1 tests H1 only. Study 2 tests H1's direction on one more domain (H2) and
  replicates Study 1 (R2). There is no Taper-plus arm, correction, replay or outcome-dependent redesign.
- **Development data only.** Both studies use development data alone; the reserved test splits stay sealed for
  any later confirmatory study.
- **What H1 covers.** It concerns the complete TaperNorm training intervention, including its taper history.
  It does not isolate a mechanism, and it does not describe normalization-free transformers in general.
- **Narrow conditions.** Results come from one small model, two domain pairs, one corpus sample per domain, one
  recipe and FP16 on T4. Generalization beyond these is untested.
- **The Chinese shift.** Under GPT-2's tokenizer most Chinese labels are byte fragments, so this shift mixes
  content with a change of the predicted token distribution, and it destroys most web performance in both
  architectures.
- **Embeddings dominate.** Embeddings are most of the parameter count, so differential embedding drift can
  mediate the effect.
- **Fragile uncertainty.** Seeds are few, and Study 2 showed that the seed-to-seed variation is much larger than
  Study 1's three seeds suggested. A screening decision does not establish a population effect.
- **Registration.** Neither study was lodged with a registry before it ran; the paper sets out exactly what
  fixes each plan in advance and what does not.

## References

- TaperNorm: [arXiv:2602.10408v1](https://arxiv.org/abs/2602.10408v1) (version 1; later versions carry a
  different title). The operator and calibration are reused; the training setup is this project's own.
- [nanoGPT](https://github.com/karpathy/nanoGPT): model starting point.
- [C4 and mC4](https://huggingface.co/datasets/allenai/c4),
  [The Stack (dedup)](https://huggingface.co/datasets/bigcode/the-stack-dedup) and
  [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) (a Study 2 candidate): data
  sources. Their own licenses and terms of use apply, and no corpus content is redistributed here.
- [OpenTimestamps](https://opentimestamps.org): the timestamp of Study 2's registration manifest.

## License

Released under the [MIT License](LICENSE). The license covers this repository's code and documents; it does
not extend to the C4, mC4, OpenWebMath or The Stack datasets, which keep their own licenses and terms of use.
