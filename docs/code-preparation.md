# Code preparation status

> Historical status for the original preparation pass. The separately authorized
> [local learning lab](local-lab.md) adds an executable toy runner and runtime
> validation; the original scientific integration prerequisites below still apply.

This pass writes code and test definitions only. No Python process, package installation, test collection, test run, corpus operation, model instantiation, analysis, benchmark, GPU rental, or training has been invoked. Source text was reviewed; runtime correctness and compatibility remain unverified.

## Prepared code

- `protocol.py`: fixed global LR/gate clocks, exact quick/full/diagnostic event lists, primary and optional run ordering with lineage and token accounting.
- `data/`: deterministic repository alias components and hash splits; original-byte content hashing; cross-domain exact duplicate precedence; five-gram candidate Jaccard; streaming 513/512 packing with document attribution and source offsets; unique-position rare counts; W/A/P/X byte classes.
- `models/`: fixed six-layer causal transformer, tied output embeddings, final RMSNorm, explicit canonical tensor copying; gamma-weighted update-level calibration EMAs, freeze/copy transition, inactive endpoint graphs and optional auxiliary target/onset.
- `training/`: AdamW parameter grouping, effective-update accumulation and once-only clipping, nonfinite detection; explicit full-state save/restore with RNG and cursors; conservative stage-wide budget admission arithmetic and safe-stop predicate.
- `evaluation/`: caller-supplied development batch evaluation with CE-only sums/counts, document/class/rare views and exact quota checks; positional sensitivity mask and energy summaries.
- `analysis/`: D/F/G-web/Q, last-five prefix slope, three-seed descriptive t interval, first-crossing matching, signed contributions and document trimming, aligned document/class decompositions, ordered pilot decisions.

These are library primitives. There is no experiment launcher, dataset download command, scheduling integration, end-to-end runner, or mechanism that interprets a draft manifest as execution approval. Calling a primitive explicitly would execute it; `execution_enabled: false` is not claimed to prevent arbitrary direct Python calls.

## Prepared tests

The synthetic tests check protocol endpoints and config schedule agreement; EOS/boundary packing and rare counts; repository transitivity and exact-dedup precedence; class edge cases; known contrasts and reconstruction; matching and decision boundaries; budget accounting; RMS parity, same-sample EMA, activation step state and zero-gate equivalence; canonical initialization, parameter counts, causal masking and auxiliary omission; and ten-update interrupted/resumed state in a small deterministic harness.

Parameter counts in test assertions are architecture-derived expectations, not measured results. The resume harness exercises persistence/RNG/EMA/Adam recovery but is not a full-transformer/data-loader resume validation. There is no claim that T09 or T10 passes.

## Explicit draft conventions requiring review before freeze

These resolve otherwise unspecified implementation details for reviewable code; they are not revisions to the supplied scientific plan or claims of published-source equivalence.

- Content hashing uses original UTF-8 bytes with no newline, whitespace or Unicode normalization. Same-split exact duplicates retain the lexicographically smallest document ID.
- Repository components hash compact UTF-8 JSON of sorted complete alias sets. Dataset-specific extraction/normalization of usable aliases remains pending actual schema inspection. C4 validation hashes the supplied stable identity modulo two; the identity selection policy must still be frozen.
- Token X treats Unicode C-category characters as non-text/control except the five explicitly permitted ASCII W bytes. Pin the Python Unicode database version with the tokenizer implementation. Empty standalone bytes are X. Review unusual non-ASCII whitespace under the supplied classification rules.
- Symmetric 1% document trimming rounds down on each end; top-contributor selection rounds up as explicitly required by the plan. Nearest matching-endpoint ties choose the earlier endpoint. A first crossing across a gap greater than 100 updates is unmatched, even if a later recrossing exists.
- Rare counts use the actual selected prefix windows sorted by source position, including unique input/label positions once. One packed source stream must identify those offsets unambiguously; the future per-seed order/materialization layer must preserve that identity.
- Budget projection applies 15% to total projected stage time, including already billed time, conservatively. The eventual watchdog requires measured checkpoint-save allowance and a reconciled session ledger.
- Complete checkpoints publish with a same-directory hard link to avoid replacing prior evidence. Filesystems without hard-link support fail explicitly. Restore is restricted by contract to trusted locally produced pickle checkpoints; checksum validation does not authenticate an untrusted producer.
- Decision input requires complete scheduled quick-dev and matching records as well as primary summary values. Missingness is explicit; an unavailable match is `None`. Prefix relative-gap calculation is not implemented: its reference denominator must be resolved and frozen before constructing decision inputs.

## Remaining integration and prerequisites

T01 and T02 remain pending: verified upstream equations/source commit/license, actual stack resolution, dataset access/schema/provenance and real revisions. No package or dataset pin is invented.

T03/T04 still need authorized dataset/tokenizer adapters, actual license/removal handling, deterministic shard sampling and replenishment, pinned MinHash-128/LSH-32x4 retrieval plus missed-candidate audits, array materialization and hashes, quota reports, per-seed orders, calibration-pool selection, and sealed test storage. Exact dedup and Jaccard helpers are not a complete near-dedup pipeline.

T05/T06 need source verification and full integration validation, paired initialization artifact hashing, branch orchestration, checkpoint rotation/durable upload, actual watchdog/session accounting and manifest admission. Performance, device compatibility and compilation behavior have not been assessed. The update primitive checks finite parameters with straightforward reductions; optimization must preserve correctness and be selected before the execution freeze.

T07 needs real dataset-backed online loaders, identity verification beyond caller-supplied batch metadata, fixed probe batches, all 12-site hooks, attention entropy, activation/gain/embedding/gradient logging and actual event/checkpoint orchestration. A batch's `split` field alone is not reserved-test isolation.

T08 needs provenance-bound loading of full/quick statistics, guardrail construction, full primary/auxiliary summary assembly, plot generation, missingness reports and archived decision outputs. Analysis primitives must not be fed mixed split/event populations. Auxiliary paired A is available mathematically from differences of D values and the uncertainty helper, but no auxiliary report pipeline exists yet.

T09/T10 remain unexecuted and their full acceptance coverage is still incomplete. T11 benchmarking, T12 freeze and T13–T20 execution/reporting remain untouched. This code-preparation pass does not bypass the protocol's dependency gates.
