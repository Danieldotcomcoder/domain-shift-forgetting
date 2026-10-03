# Environment and upstream sources — T01 pending

The scientific environment remains unresolved. The separate [laptop lab](../docs/local-lab.md)
uses a project-local environment installed by `scripts/setup-local.ps1`; its runtime
records are learning-only evidence. `pins.template.json` is an inventory to complete
for the scientific pilot, **not a lockfile**. Upstream source verification remains pending.

Draft implementation is organized as a Python source-layout package with packaging metadata in `pyproject.toml`. The core protocol/data/analysis helpers use the standard library. Model/evaluation/checkpoint helpers require the optional `torch` and `numpy` dependencies; prepared tests require `pytest`. These dependency names are not version pins or an installation instruction. Tokenizer, dataset, MinHash/LSH and plotting integrations remain unresolved. No runnable scientific entry point is provided.

At T01 record the nanoGPT commit and license, the TaperNorm paper version and equation mapping, Python and package lock, GPU driver/CUDA stack, and candidate container digest. Hardware-dependent fields must be measured on the selected machine. At T11 choose compilation and BF16 tolerances after the required checks; freeze the full stack at T12.

The protocol's external links are retained as references. Their availability, current terms, contents and compatibility have not been independently checked in these preparation-only passes. Model code is a fresh transcription of the supplied protocol, not a verified upstream adaptation. Python 3.11+ is a draft language floor, not a tested runtime claim.
