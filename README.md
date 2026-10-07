# Domain-Shift Forgetting: Does TaperNorm Forget More?

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Hardware](https://img.shields.io/badge/run%20on-Kaggle%20T4%20x2-20beff)

A pre-specified pilot experiment that asks whether replacing a transformer's internal RMSNorm layers with
**TaperNorm** (normalization that is gradually switched off during training) makes the model forget more of
what it learned when its training data switches domain.

> **H1.** After switching training from web text to Python code, does a model trained with internal
> TaperNorm (*Taper-minus*) lose more held-out web performance than an otherwise identical RMSNorm model,
> relative to simply continuing web training?

The protocol, data splits, endpoints and decision rules were fixed before any training run. This is a
screening pilot that decides whether a larger study is worth running. It is not a confirmatory result.

**Paper:** a preprint describing the study is in [`paper/main.pdf`](paper/main.pdf) (source and asset
generator in [`paper/`](paper/)).

## Result

> **H1: No.** Taper-minus did not show more persistent web forgetting than RMSNorm after the switch to Python.
> The pre-specified decision is **STOP — SMALL OBSERVED EFFECT**. Stage 1 is closed.

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
  automatically. Any new question, such as whether TaperNorm reduces forgetting, needs its own
  pre-specified protocol.

**Exploratory analyses** (chosen after the results; descriptive only, see the paper's §5.3–5.4)

- After a brief early dip, D(s) stayed near zero for most of continuation and became negative only in the final
  low-learning-rate phase (s ≥ 5,185, mean −0.020), when the three seeds also converged.
- The premise of H1 was weak in these models: the code-specific change in activation scale at the internal
  normalizer inputs was about 4% per site in both architectures, and the larger scale change in the RMS model
  happened equally when training simply continued on web text.
- A document bootstrap gives a mean-D interval of [−0.0235, −0.0211], conditional on the trained models;
  seed-to-seed variation is about 6× larger than this evaluation noise.

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

## Status

| Stage | State |
|---|---|
| Protocol v3, data specification, analysis plan | Frozen |
| Corpus preparation (C4 English + The Stack dedup Python, GPT-2 tokens) | Done, hash-verified |
| Implementation + equivalence tests | Done |
| Kaggle smoke test (T4 x2) | Passed 2026-10-03 |
| Primary run (3 seeds x 2 conditions x 3 phases) | Complete 2026-10-03 (12:18–21:40 UTC) |
| **Decision on H1** | **No: STOP — SMALL OBSERVED EFFECT. Stage 1 closed.** |

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
kaggle_h1/                 Current runner: h1_run.py, notebook builder, status script, equivalence tests
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
reports/h1-kaggle/         Decision report, frozen config, session record, run log, and per-run
                           evaluation records, training logs and completion receipts
paper/                     Preprint source, compiled PDF, and the script that generates its numbers
manifests/, registry/      Manifest templates and run/task registries
data/, artifacts/          Local data and checkpoints (contents git-ignored)
```

### Protocol documents

The original protocol pages remain the scientific source of truth:

- [Project overview](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)
- [01 · Protocol v3, H1 only](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/01%20%C2%B7%20Protocol%20v3%20%E2%80%94%20H1%20only%203d88a2d10e44812ea734fdeff5891429.md)
- [02 · Data, tokenization and leakage controls](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/02%20%C2%B7%20Data,%20tokenization%20and%20leakage%20controls%203d88a2d10e44815391dffb1e1533faa8.md)
- [03 · Analysis plan and decision rules](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/03%20%C2%B7%20Analysis%20plan%20and%20decision%20rules%203d88a2d10e4481af9a3deb8e29c432c7.md)
- [04 · Execution, GPU budget and reproducibility](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/04%20%C2%B7%20Execution,%20GPU%20budget%20and%20reproducibility%203d88a2d10e44811ebd95cc0afce540c9.md)
- [05 · Decision sheet, limitations and revision record](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/05%20%C2%B7%20Decision%20sheet,%20limitations%20and%20revision%20reco%203d88a2d10e4481298fd9f24f8a9b7309.md)

## Scope and limitations

- **One question only.** H1 is the only hypothesis. There is no Taper-plus arm, correction, replay, extra seeds
  or outcome-dependent redesign.
- **Development data only.** The pilot uses development data alone; the reserved test split stays sealed for any
  later confirmatory study.
- **What H1 covers.** It concerns the complete TaperNorm training intervention, including its taper history.
  It does not isolate a mechanism, and it does not describe normalization-free transformers in general.
- **Narrow conditions.** Results come from one small model, one corpus sample, one recipe and FP16 on T4.
  Generalization beyond these is untested.
- **Embeddings dominate.** Embeddings are most of the parameter count, so differential embedding drift can
  mediate the effect.
- **Fragile uncertainty.** With three seeds, uncertainty estimates are fragile, and a positive screen does not
  establish a population effect.

## References

- TaperNorm: [arXiv:2602.10408v1](https://arxiv.org/abs/2602.10408v1) (version 1; later versions carry a
  different title). The operator and calibration are reused; the training setup is this project's own.
- [nanoGPT](https://github.com/karpathy/nanoGPT): model starting point.
- [C4](https://huggingface.co/datasets/allenai/c4) and
  [The Stack (dedup)](https://huggingface.co/datasets/bigcode/the-stack-dedup): data sources. Their own
  licenses and terms of use apply, and no corpus content is redistributed here.

## License

Released under the [MIT License](LICENSE). The license covers this repository's code and documents; it does
not extend to the C4 or The Stack datasets, which keep their own licenses and terms of use.
