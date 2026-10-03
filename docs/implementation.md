# Implementation requirements and task traceability

The imported source task statuses remain unchanged. Draft primitives and test cases are now present; their precise coverage and gaps are recorded in `code-preparation.md`. No task is certified complete: T01/T02 prerequisites remain unresolved, T03–T08 lack validated integration, T09/T10 have not run, and T12 is not frozen. The individual source task pages retain the exact acceptance criteria and dependency graph.

## T01–T04: sources and data

T01 → `environment/`, model equation mapping, package/source lock evidence. T02 → data access/schema evidence, actual revisions and terms, repository alias availability. Neither depends on a GPU rental. T03 depends on T02; T04 depends on T03.

Data contracts must preserve dataset revision, shard/row, canonical content hash, all available repository aliases or URL, license provenance, split and selected token spans. Define canonicalization, shard ordering, bounded shuffle policy, group identity serialization and C4 50/50 hash convention explicitly before data freeze. These implementation details are not supplied completely and must not be silently invented as scientific facts.

Rebuild connected repository groups across the whole expanded pool before freezing. Reject missing usable code provenance. Apply global exact dedup and the specified approximate near-dedup audit; retain test over dev, remove matching train, replenish deterministically and repeat the audit. Record counts, truncation and missed-candidate checks. Never repeat documents to satisfy quotas.

513-token windows at stride 512 must expose 512 labels exactly once with aligned document IDs; EOS belongs to the preceding document. Remainders continue within a split only. Quick-dev is a fixed subset of full dev. Preserve the GPT-2 byte mapping and literal-special-text policy. Class X takes precedence, then W/A/P; R overlaps the disjoint classes and counts unique prefix stream positions once. Store all 50,257 token classes and per-seed order hashes. Calibration audit pools are training-only and identical across conditions.

## T05–T06: model and training

T05 depends on T01; T06 depends on T05. Future model files should separately own transformer topology, RMS/Taper operators and canonical initialization. Map protocol §Internal TaperNorm to its forward operator, update-level gamma-weighted EMA accumulators, bias correction and freeze transition. Verify against the published source during T01; this pass only transcribes the supplied protocol.

For every internal site, use `r(h)=sqrt(mean(h²)+1e-6)` and `y=g*(h/r(h))*gamma+(1-g)*c*h*gamma_tilde`. Accumulate numerator `mean(||h*gamma||²/r(h))` and denominator `mean(||h*gamma||²)` across the effective update using pre-step values; update EMAs once. Freeze `c=EMA_bc(numerator)/(EMA_bc(denominator)+1e-12)` after update 763 and copy then-current gamma. Activate gamma-tilde's first Adam step at 764. Skip inactive branches and parameter steps; gamma becomes inactive at zero gate. Do not fold the training parameterization. Final RMSNorm stays active.

Taper-plus alone freezes the warmup pre-final residual RMS target and activates coefficient 0.1 at 764. Evaluation always excludes auxiliary loss. Record this convention and endpoint optimizer behavior as study choices, not exact replication claims.

Copy canonical initialization tensors; RNG seed equality alone is insufficient. At update 9,156 branch each condition's own immutable full state into web and Python. Preserve optimizer moments and per-parameter steps, RNG, data cursor/order and global LR/gate clocks. Do not reset optimization at the switch. Clip once after accumulation. Full-state contracts are in `docs/artifact-contracts.md`.

## T07–T10: evaluation, analysis and correctness

T07 depends on T04 and T05; T08 on T07; T09 on T05 and T06; T10 on T04 and T08. Exact acceptance cases are in `tests/README.md`.

Expand schedule unions into sorted, deduplicated event lists; preserve both global and continuation update coordinates. At s=0, branch evaluations refer to the same prefix state within a condition. Store FP32 weight snapshots and CE sums/counts by document and disjoint class at every full-dev point. Quick-dev values never enter the full-dev primary contrast.

Analysis implements the source's D/F/G-web/Q identities, chronological matching with brackets no longer than 100 updates, tail decompositions, all seed values, sample SD and descriptive fixed-n interval. Missing data remain missing. Apply the seven ordered decision categories exactly; common matched target absence does not become a numeric zero. Retain all failed and unfavorable observations.

Diagnostics cover all 12 internal sites, source/code energies and log-k, norm p99/p50, RMS pre-norm energy, embedding norms by class/R, activation RMS, branch/residual ratios, gains, gradients and clipping. Store all-position and exclusion-mask statistics separately; exclude positions 0–15 and the first token after EOS only for the sensitivity view. Attention entropy uses a separate fixed forward probe. These observations do not identify a correction mechanism.

## T11–T12: feasibility and freeze

T11 depends on T09 and T10. Benchmark real data by phase, quick/full evaluation, diagnostics, checkpoint I/O and compilation startup. Choose compiled/uncompiled globally and primary-only or the full auxiliary trio before outcomes. The projected total times 1.15 must fit 24 billed GPU-hours, including all prior attempts and setup. If infeasible, stop feasibility; do not delete controls or resize the model.

T12 depends on T11. Resolve the [freeze checklist](freeze-checklist.md); a draft template cannot authorize seed 101. Any watchdog must account for the entire allocation, not a fresh 24 hours per run, and reserve enough time to save state safely.

## T13–T20: deferred execution and reporting

T13 → T14 → T15 execute primary seeds 101 → 102 → 103 after T12. Within each seed: RMS prefix, web, Python, then Taper-minus prefix, web, Python. T16 → T17 → T18 follow T15 only if the full auxiliary allocation was frozen. T19 depends on T15; auxiliary reporting waits for completion or documented termination if allocated. T20 depends on T19 and archives every attempt and permitted pilot conclusion. No next-stage experiment is launched by the decision.
