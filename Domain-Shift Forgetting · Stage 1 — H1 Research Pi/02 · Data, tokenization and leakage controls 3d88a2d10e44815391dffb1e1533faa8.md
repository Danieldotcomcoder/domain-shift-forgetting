# 02 · Data, tokenization and leakage controls

[Domain-Shift Forgetting · Stage 1 — H1 Research Pilot](../Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)

## Exact inputs

Web: allenai/c4, configuration en, official train split, field text. Code: bigcode/the-stack-dedup, Python files under data/python, train split, field content. Preserve comments, docstrings, indentation and original text. This is web-to-Python, not a multilingual-code experiment.

Reserve 260,000,000 web training tokens and 110,000,000 Python training tokens. For each domain prepare 2,097,152 supervised full-development labels and 8,388,608 reserved-test labels; quick-dev is a fixed 262,144-label subset of full dev. Counts refer to GPT-2 tokenization, not words.

Use corpus sampling seed 20260911, independent of model seeds. Pin actual revisions after access is verified. Do not invent commit hashes. Stream for preparation only; train from materialized local arrays. Record deterministic shard order and any bounded shuffling algorithm/buffer before sampling.

## Access and provenance gate

Verify authorized access to the code dataset and its current usable version before renting a GPU. Respect original license attribution and maintainer removal requirements; if removals force a revision, document the new corpus identity rather than silently changing data.

Manifest each selected document: dataset revision, shard/row identity, canonical content SHA-256, repository aliases or web URL, split, selected token spans and license provenance where supplied.

Do not download full corpora. Keep preparing deterministic additional shards until all post-filter token quotas are met. Never silently repeat training documents to reach a quota.

## Split construction

Use official C4 train for training. Deterministically hash official C4 validation document IDs/content hashes into 50% development and 50% reserved test. Remove matching training documents; resolve duplicate dev/test documents before finalizing.

For code, use all repository-name aliases available in the actual schema. Build connected groups over the selected pool for files sharing a repository alias. Drop files with missing usable provenance. Assign groups by SHA-256 group identity, modulo 100: 0–89 train, 90–94 dev, 95–99 test. These percentages apply to groups, not tokens.

After expanding the candidate pool, recompute groups and assignments before freezing; never extend a frozen pool with a new group connection unnoticed. Repository names are an incomplete proxy for forks. Report that limitation.

Exact dedup: canonical content hashes across all domains and splits. Near-dedup audit: fixed GPT-2 token 5-grams, MinHash 128 permutations, fixed seed 20260911, LSH candidate retrieval followed by exact candidate Jaccard ≥0.85. Pin LSH bands/rows in the implementation manifest; use 32 bands ×4 rows as the starting fixed setting. This is an approximate audit, not a guarantee that every near duplicate is found.

Remove training documents matching either held-out split. For dev/test overlap retain test and drop dev, replenish deterministically, then rerun the audit. Record before/after counts and sampled missed-candidate checks. No token arrays may cross split boundaries.

## Packing and loss attribution

GPT-2 BPE vocabulary 50,257, EOS 50,256. Pin tokenizer files and package versions. Literal special-token-looking text is ordinary text; append one actual EOS after each document.

Use windows of 513 tokens at stride 512: positions 0–511 are inputs and 1–512 are labels. Overlap only the boundary context token; each supervised target is counted once. Continue remainder buffers across documents within the same split. Permit causal attention across EOS, reset learned positions each window, and use no padding targets. Log this contextualization policy.

Maintain a label-aligned original document-ID array through packing. Attribute EOS to its preceding document; a target after EOS belongs to the new document even if its context includes the previous one. Store per-document token counts after packing, not raw-text lengths. This makes token-weighted document aggregation exactly reconstruct corpus CE.

Discard only terminal incomplete windows, and record losses of available tokens to truncation. Select whole deterministic packed windows to get exact dev counts. Same evaluation windows, labels and weights across all conditions.

Generate training orders per seed 101–103 once, then reuse within that paired group. Shared corpora across seeds imply uncertainty is conditional on the selected corpus, not population-wide sampling uncertainty.

## Token classes

Decode token bytes using the pinned GPT-2 byte mapping. Use disjoint classes: W = nonempty bytes containing only ASCII space/tab/newline/CR/form-feed; A = valid decoded text containing a Unicode alphanumeric character; P = remaining valid non-control punctuation/symbol tokens; X = special EOS, other controls or invalid standalone UTF-8 byte fragments. Resolve X before the others. Publish the complete 50,257-ID table.

This explicit X class prevents control bytes and EOS being mislabeled as ordinary punctuation. Main non-whitespace code CE excludes only W and therefore includes X; separately report A+P-only code CE as sensitivity analysis.

Rare flag R: fewer than 100 occurrences in a seed's actual web prefix inputs/labels, using one documented unique stream count convention. Use the prefix source stream's token positions counted once. R overlaps W/A/P/X and is not added as a fifth disjoint contribution.

Rare input occurrence does not imply untrained tied embeddings: the output softmax updates shared rows too.

Report token counts and proportions for all classes/splits; do not assume whitespace dominates observed loss without measurement.

## Calibration audit pools

At the switch use 131,072 web-training and 131,072 code-training supervised-token positions, selected deterministically from separate training-pool documents where possible. They must never come from dev or test. All conditions receive identical forward-only exposure. Log these passes separately from gradient-bearing tokens.

## Gate 0 acceptance

- [ ]  Readable sample and actual schema recorded.
- [ ]  Revisions, license/provenance fields and tokenizer hashes recorded.
- [ ]  Post-filter unique token quotas satisfied without per-trajectory repetition.
- [ ]  Train/dev/test document/group integrity and duplicate audit passed.
- [ ]  Packed labels, document attribution and class table validated.
- [ ]  Array hashes and per-seed orders frozen.
- [ ]  Reserved test data excluded from online loaders and dashboards.

## Sources

[C4 dataset documentation](https://huggingface.co/datasets/allenai/c4). [The Stack dedup dataset documentation](https://huggingface.co/datasets/bigcode/the-stack-dedup). The split, quota and audit choices above are this project's protocol, not guarantees supplied by those datasets.