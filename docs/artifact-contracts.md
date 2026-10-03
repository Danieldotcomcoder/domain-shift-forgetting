# Artifact contracts for future implementation

These are design requirements, not validated serializers. Future code must version and validate records and reject incomplete frozen manifests. Null denotes unavailable evidence; an empty count must not be used as a loss value.

## Corpus and arrays

Each selected document needs a stable ID, dataset/config/revision, shard and row, canonical SHA-256, URL/repository aliases, group and split, license provenance where supplied, selected token spans, packed scored-token counts, and removal/audit lineage. Store authorized text and potentially sensitive provenance locally, outside public reports.

Every packed-array manifest needs domain/split, dtype/shape, tokenizer file hashes, window/stride, token and label counts, EOS and contextualization rules, aligned document-ID array hashes, truncation counts, class-table hash, order hashes and quick-to-full subset mapping. Record token spans explicitly enough to reproduce rare-prefix counts without boundary duplication. Test-array references must not enter online loader configuration.

## Evaluation sufficient statistics

An evaluation event needs schema version, run/attempt ID, frozen manifest hash, checkpoint hash, global update, continuation update if applicable, domain, quick/full split role, evaluation array hash, CE sum and scored-label count. Derive CE by sum/count; never average batch means without weights.

Document/class rows additionally carry document ID or W/A/P/X, fixed label count and CE sum. Keep R as a separate overlapping view. Preserve intersections if needed to audit reconstruction. Record unavailable subgroup CE as null with a reason. Forward diagnostics record probe hash, mask, site, statistic, counts and any near-zero-energy flag separately from CE and gradient-bearing tokens.

## Complete checkpoints

Complete state includes schema/version, weights, optimizer groups/moments/per-parameter steps, any scaler, calibration buffers/counters, frozen c and auxiliary target, gain activation state, completed global updates, LR/gate state, all RNG state, data cursors and order hashes, code/config/manifest references. Switch-state hashes link both children to their own condition's prefix. Resume must recover a specific attempt and checkpoint, never silently create a substitute seed.

Weight-only full-dev snapshots are not resumable checkpoints. Complete states are also saved at branch endpoints and rotated every 15 minutes. Durable prefix URI and verified checksum are required before rental teardown.

## Attempts and budget

Each actual attempt records run and parent lineage, start/end UTC, completion/outcome or failure reason, updates/tokens completed, hardware/stack, manifest and checkpoint hashes, billed session reference and attributable cost/time. Sessions record disjoint billed intervals and all setup/failure/idle overhead. Reconcile attempt allocations with session totals rather than adding both totals. All consumed GPU time counts against the single stage cap.

## Analysis outputs

Bind outputs to the frozen analysis code/config and exact input hashes. Store per-seed prefix gap/adaptation guardrails, every F, D at 1,525/3,050/6,104, G-web/Q, all matching targets and brackets, sensitivity endpoints, signed document contributions, class contributions and seed uncertainty. Preserve actual versus interpolated and full versus quick labels. The ordered decision includes missingness and limitation reasons and never launches more work.
