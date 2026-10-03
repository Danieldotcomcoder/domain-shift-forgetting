# Scientific acceptance specification

Synthetic tests in `test_*.py` cover protocol schedules, data primitives, analysis,
norm/EMA behavior, model pairing/causality, and a small ten-update checkpoint
harness. The separate [local lab](../docs/local-lab.md) adds training, branch,
and checkpoint replay tests. These cover only part of the acceptance requirements
below; full-model integration, accumulation, full diagnostic hooks and operational
checks still require work. Torch-dependent tests skip when the model environment
is absent; a skipped test is never gate evidence. Required scientific evidence must
contain code/config hashes, exact inputs, expected outcomes, measured errors,
environment and pass/fail status. An unchecked requirement is not a pass.

## Model and resume (T09)

- FP32 RMS endpoint and separate folded-inference tensor/logit equivalence: rtol 1e-5, atol 1e-6; record max errors. BF16 tolerances are separately justified after FP32 passes.
- Both active branches receive gradients. Gamma-tilde has no Adam state before activation, first steps at 764 and changes afterward. Gamma retains its existing state until inactive at gate zero. Inactive parameters do not decay or step.
- Gate/calibration boundary checks at 763, 764, 6103 and 6104; zero gate performs no internal token-dependent division; final RMSNorm remains present.
- EMA once per effective update matches offline reconstruction of the same recorded pre-step update means, with bias correction, FP32 rtol 1e-5. Validate c freeze and gamma copy ordering.
- Auxiliary target frozen after warmup, coefficient active only from 764, and auxiliary loss absent from every evaluation CE.
- Canonical initial tensors copied across conditions, zero gain decay, residual initialization and actual parameter counts checked.
- Deterministic microbatch accumulation equivalence and single global clipping after accumulation.
- Ten uninterrupted versus resumed updates: compare weights, optimizer moments/steps/groups, LR/gate, EMA, RNG and data cursor. Do not assert bitwise equivalence for nondeterministic kernels.
- Web and Python children load the identical full prefix state within each condition; global LR clock does not restart.

## Data and analysis (T10)

- Repository alias transitivity, missing provenance rejection, frozen-pool connection handling, split isolation and train-heldout dedup precedence.
- No cross-split packing, each target counted once, EOS document attribution and after-EOS target attribution, terminal truncation accounting, exact quotas and fixed quick/full subset mapping.
- Literal special-looking text handled as ordinary text; complete W/A/P/X partition including invalid UTF-8/control cases; R overlaps without double counting and uses unique prefix stream positions.
- Online loaders and dashboards cannot resolve reserved-test data; calibration/entropy probes come only from specified permitted pools.
- Four identical branches yield D=0; known synthetic shifts recover D/F/G-web/Q and their identity.
- Weighted document and W/A/P/X contributions reconstruct corpus D within 1e-6 nats; R never added as a fifth disjoint contribution.
- Signed top-contributor ranking, ceil(1% count), document-ID ties, trimmed mean, positive-mass fraction and near-zero net-D handling match the source.
- Matching covers earliest exact match, first crossing, flat and nonmonotonic curves, unreachable/already-surpassed targets, brackets above 100 updates and common-target absence; no extrapolation or later favorable recrossing.
- Decision precedence covers missing seed/failure, guardrails, equality at -0.03/0.015/0.03, strict D/Q/matched requirements, transient common timestamps and missing auxiliary runs.
- Full/quick schedules have s=0 and exact endpoints, sorted unique events and correct global coordinates. Primary calculations cannot consume quick-dev data.
- Plots distinguish observed/interpolated points and full/quick-dev; ratios and uncertainty labels preserve stated limitations.

## Operational checks before freeze

Validate full-state persistence, durable checksum verification, safe watchdog termination with checkpoint-save allowance, cumulative billed-session accounting including failures, and admission of only complete frozen manifests. These checks remain future work.
