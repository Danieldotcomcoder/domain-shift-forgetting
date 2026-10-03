# Research structure and artifact ownership

This repository scaffold separates the scientific specification, future implementation, local corpus storage, and empirical evidence. It introduces no new scientific condition or experiment. The supplied export stays unchanged; `source-index.md` locates it and `manifests/source-documents.json` inventories its file hashes.

## Configuration and freeze

`configs/stage1.v3.json` transcribes protocol constants with their update/token units. It is deliberately not runnable. Nulls identify unresolved facts or choices. The complete supplied documents govern details not encoded in JSON. Discrepancies must be reconciled against them before implementation is frozen; do not silently substitute defaults.

`environment/` owns actual dependency and upstream-source pins when established. `manifests/templates/` owns record templates. `manifests/frozen/` is reserved for timestamped, immutable T12 snapshots containing all required hashes and resolved choices; it is currently empty except for documentation.

## Package responsibilities

- `src/domain_shift_forgetting/data/`: access/provenance, repository connected groups, canonical hashes, cross-domain dedup, deterministic packing, document attribution, token classes, training orders, calibration pools. Preparation is distinct from online loading.
- `src/domain_shift_forgetting/models/`: paired canonical base tensors, transformer, internal RMS/Taper operators, final RMSNorm, calibration buffers and auxiliary target. No optimizer or data-loading side effects.
- `src/domain_shift_forgetting/training/`: effective-update accounting, optimizer state semantics, global LR/gate clock, immutable switch branching, complete resume, attempt accounting and safe budget watchdog.
- `src/domain_shift_forgetting/evaluation/`: fixed quick/full-dev event schedules, CE-only sums/counts, document/class statistics and forward-only diagnostics. Access only authorized training probe pools and development arrays.
- `src/domain_shift_forgetting/analysis/`: deterministic D/F/G-web/Q, matching, tails, seed uncertainty and ordered decisions from stored statistics. No training, checkpoint selection, or test-set access.

These are the intended full module responsibilities. Draft APIs now cover a subset; see `code-preparation.md` for the current implementation boundary. No primitive returns fabricated measurements. Missing adapters and orchestration are explicit gaps, not stubs that pretend to succeed.

## Storage and lineage

`data/raw/` holds authorized preparation inputs; `data/interim/` holds provenance, grouping and dedup intermediates. `data/processed/` holds training/dev arrays and label-aligned metadata. `data/reserved_test/` is a separate storage boundary for sealed test artifacts. A directory name alone does not enforce isolation: T10 must prove online loaders and dashboards cannot access it.

`registry/planned-runs.csv` preserves the 27 source run records: 18 mandatory primary records and 9 conditional auxiliary records. A record is a prefix or continuation, not an independent seed. `registry/tasks.csv` preserves T01–T20 and their dependencies. These are local starting copies, not a synchronization integration with Notion.

`registry/attempts.csv` will contain one row per actual attempt, including failures and resumes. `registry/gpu-sessions.csv` will account for disjoint billed rental intervals, preventing double-counting shared setup time. Neither ledger currently contains observations.

`artifacts/checkpoints/` holds local states; `artifacts/logs/` holds events; `artifacts/validation/` holds actual gate evidence. Prefix states require durable external copies and verified checksums before rental termination. `reports/` holds later derived summaries, figures and archive indexes, with manifest and input hashes. Large/local artifacts are ignored by Git; small provenance records remain trackable. Ignore rules do not provide access control or backups.
