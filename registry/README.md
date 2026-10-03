# Planning records and evidence ledgers

`tasks.csv` and `planned-runs.csv` are copied from the supplied export, preserving original status, dependencies and blank evidence. They do not certify completion. `planned-runs.csv` includes the optional auxiliary records as Conditional, not allocated or scheduled. Original Notion-export pages remain authoritative reference documents; these CSV copies have no live synchronization.

`attempts.csv` and `gpu-sessions.csv` are header-only templates with no observed records. Populate only after actual activity. Retain failures and infrastructure resumes; link attempts to planned run IDs and frozen manifests. Empty measurements mean pending. Use one stage-wide billed-time total and reconcile shared overhead according to `docs/artifact-contracts.md`.
