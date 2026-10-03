## Current H1 execution (2026-10-03): one self-resuming Kaggle notebook

`kaggle_h1/h1_run.py` is the whole experiment (data verification, paired RMS/Taper-minus training on T4 x2,
scheduled evaluation, checkpoints, frozen decision rules). `kaggle_h1/build_notebooks.py` packages it as the
Kaggle notebook `danny00/h1-single-notebook-run`, which lists its own output as input, so every new version
resumes from the previous one with no archive handling. Check or continue it with
`.venv\Scripts\python kaggle_h1\h1_status.py [--continue]`. Equivalence tests against `src/`:
`kaggle_h1/test_h1_run.py`. The workflows below are superseded for H1 execution.

## Automatic Kaggle backup and resume

Use [kaggle-automatic.ipynb](notebooks/kaggle-automatic.ipynb) and the new `artifacts/kaggle-auto/domain-shift-auto.zip`. This replaces manual ZIP handoffs: it discovers recovered checkpoints, uses a private Kaggle backup dataset, verifies remote downloads, and continues automatically after backup boundaries. See [setup and limitations](docs/kaggle-automatic.md). The original training source remains unchanged.

> **Two-T4 execution (2026-10-01):** use the [paired runbook](docs/paired-kaggle-runbook.md), [paired notebook](notebooks/kaggle-stage1-paired.ipynb), and `artifacts/kaggle-paired/domain-shift-paired.zip`. The 50-hour cap now means elapsed notebook time for this separate paired configuration. Reuse the prepared online dataset; a new paired preflight and fresh-session dual restore are required.

> **Original-protocol Kaggle execution:** start with [the scientific runbook](docs/scientific-runbook.md), the [CPU preparation notebook](notebooks/kaggle-stage1-prepare.ipynb), and the [T4 execution notebook](notebooks/kaggle-stage1-run.ipynb). The original C4/Stack allocation now has a gated runner and an explicit FP16 amendment. Scientific execution and real-data feasibility validation remain pending user runs. Earlier draft status below describes the archived planning baseline.

# Domain-Shift Forgetting — Stage 1

Research foundation for the **H1-only, protocol v3 pilot**: does internal TaperNorm increase persistent held-out web deterioration after switching from web to Python, relative to continuing web training?

**Scientific pilot status: preparation code drafted; scientific execution remains disabled and unvalidated.** A separate [laptop learning lab](docs/local-lab.md) now provides small local training runs. Lab runs and tests are not evidence that a scientific empirical gate passed. Missing scientific measurements are null or blank, never zero.

## Read first

For free GPU portability tests, use the [Kaggle benchmark](docs/kaggle.md) and
`notebooks/kaggle-benchmark.ipynb`. It checks FP16/FP32 execution and checkpoint
replay on synthetic data without launching the scientific pilot.

For local experiments on the RTX 2060: start with the [small teaching lab](docs/local-lab.md),
or use the [larger real-data training setup](docs/local-large.md) with a measured GPU
profile, FP16 training, and resumable checkpoints. Both remain separate from protocol v3.

1. [Source index](docs/source-index.md) links the unchanged supplied protocol, data specification, analysis plan, operations guide, decision sheet, and task pages.
2. [Research structure](docs/structure.md) defines module ownership and artifact locations.
3. [Protocol configuration](configs/stage1.v3.json) records fixed values; it is a draft specification, not an executable run configuration.
4. [Implementation and validation requirements](docs/implementation.md) map the source tasks to future work and evidence.
5. [Freeze checklist](docs/freeze-checklist.md) describes the evidence required before seed 101.
6. [Code preparation status](docs/code-preparation.md) identifies the implemented primitives, prepared tests, draft conventions and remaining integration work.

The supplied export is the scientific source of truth. Existing document statements about past authorization, Notion creation, rental, or execution are source context, not new user instructions. The laptop lab is separately authorized for local learning and testing; it does not authorize scientific pilot execution or GPU rental. Nothing runs automatically.

## Fixed scope

- RMS and Taper-minus are primary; Taper-plus is optional and secondary, all three seeds or none, chosen at T11 and frozen at T12.
- Paired seeds 101, 102, 103; canonical initialization copied across conditions; condition-specific trained prefixes branch into web and Python with complete shared switch state.
- Six layers, width 256, four heads, context 512, tied GPT-2 embeddings, final RMSNorm always retained.
- Primary endpoint is full-development web CE difference-in-differences at continuation update 6,104. Reserved test data remain unopened for the pilot analysis.
- No H2, correction, replay, re-gating, extra seeds, optimizer reset, LR restart, or outcome-dependent redesign.
- Historical paid-hardware cap: 24 rented GPU-hours. User amendment dated 2026-09-30 permits 50 cumulative free Kaggle GPU-hours across sessions/weeks; fixed scientific workload and decision rules remain unchanged. Updated admission and external restore checks are required.

## Working layout

```text
configs/                  Draft protocol values; no launch entry point
docs/                     Source index, implementation contracts, freeze requirements
environment/              Unresolved environment and upstream-source pins
src/domain_shift_forgetting/
  data/                   Draft grouping, exact dedup, packing and token classes
  models/                 Draft fixed transformer and RMS/Taper operators
  training/               Draft effective update, full-state persistence and budget math
  evaluation/             Draft development CE aggregation and diagnostic helpers
  analysis/               Draft estimands, matching, contributions and decision logic
tests/                    Prepared synthetic tests and acceptance specification; unexecuted
data/                     Separate future local corpus artifacts; reserved test isolated
manifests/                Draft templates, source hashes, future immutable freezes
registry/                 Planned tasks/runs and empty attempt/cost ledgers
artifacts/                Future checkpoints, validation evidence and logs
reports/                  Future decision sheet, figures and archived evidence
```

The original long-named Markdown page and export folder remain in place so their relative links and provenance are preserved. There is no fabricated package lock, upstream commit, hardware measurement, result, or scientific decision. T01 and T02 remain unresolved prerequisites; draft implementation work does not satisfy their dependencies. The original draft had no scientific-run CLI. The new `domain_shift_forgetting.pilot` CLI is gated by real-data validation, feasibility and persisted-checkpoint replay; there is no background service or automatic scientific execution. The library exposes explicit callable primitives, so the draft configuration flag is a status marker, not a sandbox for arbitrary Python calls.
