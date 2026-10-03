# 05 · Decision sheet, limitations and revision record

[Domain-Shift Forgetting · Stage 1 — H1 Research Pilot](../Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)

## Current state

Protocol: v3 prepared, not yet execution-locked. Dataset access: pending. Implementation: not started. Benchmark: pending. Primary seeds completed: 0/3. Measured GPU-hours: not recorded. Scientific decision: pending.

## Final decision sheet — fill after execution

- Allocation chosen before outcomes: pending.
- Freeze timestamp and manifest/code/analysis hashes: pending.
- Gate 0 data integrity evidence: pending.
- Correctness tests and tolerances: pending.
- GPU phases, end-to-end projection and actual billed time: pending.
- Per seed: pre-switch web/code CE, relative prefix gap, non-whitespace improvement, F for all four branches, D at 1525/3050/6104, G-web, Q, common adaptation targets and matched differences: pending.
- Seed mean, SD and descriptive uncertainty interval: pending.
- Tail concentration, trimmed/median statistics and token-class contributions: pending.
- Zero-shot energies/dispersion, embedding norms and clipping: pending.
- Auxiliary availability and paired exploratory A contrast: pending/not scheduled.
- Missing/invalid runs and all attempted fixes: pending.
- Exact ordered classification and permitted wording: pending.
- Next action and remaining resources: pending.

## Required plots

1. Full-dev web CE versus continuation tokens for each branch, with seed-level curves.
2. D(s) per seed and mean; label quick-dev versus full-dev.
3. Actual forgetting and Q/G-web decomposition.
4. Web forgetting versus non-whitespace code CE with observed points and matching brackets.
5. Sorted signed document contributions and token-class contributions.
6. Per-site energy shift and activation dispersion at the switch.

## Revision record: v2 → v3

- H1-only scope confirmed; all Stage 2 intervention schedules removed from executable work.
- Fixed three primary seeds; removed outcome-dependent early stopping and extra seeds.
- Optional auxiliary condition now all three seeds or none, selected by cost before outcomes.
- Fixed ~18M model; no performance-dependent larger architecture.
- G-web replaces the causal-bound reading of R-est; exact Q decomposition retained.
- No assertion that shared embedding drift only dilutes D, or that rare tied embeddings stay near initialization.
- Gain initialization/freeze contradiction repaired; optimizer inactive-state and aux onset explicitly specified.
- EMA equality test uses identical recorded samples; snapshot comparison is descriptive.
- Explicit s=0/1525/3050 endpoints, deterministic first-crossing matching and no slope ratio near zero.
- Control/special/invalid-byte token class added; overlapping rare flags cannot be double-counted.
- Decision categories ordered, uncertainty distinguished from investment thresholds; no unsupported negative or mechanism conclusion.
- Realistic storage allowance and budget accounting separate optimizer work from evaluation overhead.

## What a positive pilot would mean

The complete internal-taper training recipe shows persistent differential deterioration in this controlled small-model setting and merits a newly designed follow-up. It would not establish that calibration mismatch caused forgetting, that all norm-free models fail, or that a correction works.

## What a small or inconclusive pilot would mean

Evidence is insufficient to justify the next investment under this protocol. A small mean or wide interval is not proof that the effect is absent. Report the observed interval and finite-budget/model/data limitations.

## Deferred research questions

No Stage 2 is authorized by this protocol. A later design must address optimizer-state transformations, epsilon, global clipping, matched adaptation and fair optimization controls. Re-gating failure cannot rule out every scaling intervention. Replay success does not erase potential value where source data retention is unavailable. Uniform near-one energy ratios do not mathematically prove a multilayer intervention has no effect.

If auxiliary protection appears, a later factorial RMS-plus/Taper-plus study can test whether the auxiliary benefit is specific to tapering. Outlier concentration motivates investigation but does not predetermine the mechanism.

## Research provenance

Source specification: user-provided [domain-shift-stage1-H1-plan-v2.md](http://domain-shift-stage1-H1-plan-v2.md), reviewed against the prior starter plan. This project is the corrected successor, not an executed result.

[TaperNorm paper](https://arxiv.org/html/2602.10408v1). [LayerNorm-removal paper and rare-token tail observations](https://arxiv.org/html/2507.02559v1). Neither establishes the proposed H1 result for this exact web-to-Python training experiment.

A new literature/novelty audit, power analysis and independent confirmation remain necessary before publication claims. This project is an execution plan, not a guarantee of novelty or acceptance.