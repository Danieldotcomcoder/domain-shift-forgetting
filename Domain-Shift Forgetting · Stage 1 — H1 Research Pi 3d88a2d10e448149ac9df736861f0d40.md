# Domain-Shift Forgetting · Stage 1 — H1 Research Pilot

<aside>
🔬

**H1 only:** Does internal TaperNorm increase persistent web deterioration after switching training from web text to Python?

Status: protocol prepared; implementation, data checks and training not started. Version 3 supersedes the uploaded v2 for this project.

</aside>

## Start here

Read the protocol, complete data and correctness gates, benchmark one GPU, and freeze the execution manifest before seed 101. The experiment has a hard 24 GPU-hour cap, including failures. A pilot decision is not a confirmatory research claim.

## Project navigation

- [01 · Protocol v3 — H1 only](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/01%20%C2%B7%20Protocol%20v3%20%E2%80%94%20H1%20only%203d88a2d10e44812ea734fdeff5891429.md)
- [02 · Data, tokenization and leakage controls](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/02%20%C2%B7%20Data,%20tokenization%20and%20leakage%20controls%203d88a2d10e44815391dffb1e1533faa8.md)
- [03 · Analysis plan and decision rules](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/03%20%C2%B7%20Analysis%20plan%20and%20decision%20rules%203d88a2d10e4481af9a3deb8e29c432c7.md)
- [04 · Execution, GPU budget and reproducibility](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/04%20%C2%B7%20Execution,%20GPU%20budget%20and%20reproducibility%203d88a2d10e44811ebd95cc0afce540c9.md)
- [05 · Decision sheet, limitations and revision record](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/05%20%C2%B7%20Decision%20sheet,%20limitations%20and%20revision%20reco%203d88a2d10e4481298fd9f24f8a9b7309.md)
- [Implementation tasks](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks%20facc2f8abdc3499a99aa523d288eaadf.md)
- [Run registry](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Run%20registry%20310a761cff234f6ab5b3564877a7e102.md)

## Scope

Primary: RMSNorm versus Internal-Taper without auxiliary loss. Secondary: Internal-Taper with auxiliary loss. Final RMSNorm retained throughout. No correction, replay, re-gating or H2 intervention is scheduled in Stage 1.

## First action

Start with [T01 · Pin environment and implementation sources](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks/T01%20%C2%B7%20Pin%20environment%20and%20implementation%20sources%203d88a2d10e4481f6b3bec60f1de2f8d6.md) and [T02 · Verify dataset access and inspect provenance schema](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks/T02%20%C2%B7%20Verify%20dataset%20access%20and%20inspect%20provenance%203d88a2d10e4481f7b617debd9787c1c4.md). Both precede paid training.

After implementation and validation, [T11 · Benchmark GPU and lock feasible allocation](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks/T11%20%C2%B7%20Benchmark%20GPU%20and%20lock%20feasible%20allocation%203d88a2d10e4481e9b131cd4a0ae65bf7.md) determines the feasible allocation; [T12 · Freeze execution manifest and analysis version](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks/T12%20%C2%B7%20Freeze%20execution%20manifest%20and%20analysis%20versi%203d88a2d10e44810ab981f924326214f3.md) locks it before results.

The model is fixed at approximately 18M parameters with three primary seeds. Primary allocation: 2.10B training tokens; full optional auxiliary allocation: 3.15B. No outcome-based extra seeds. The operations guide contains the dated GPU rental comparison.

## Evidence status

No training results, throughput measurements or costs incurred are recorded. Empty measurements mean pending, never zero.

[01 · Protocol v3 — H1 only](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/01%20%C2%B7%20Protocol%20v3%20%E2%80%94%20H1%20only%203d88a2d10e44812ea734fdeff5891429.md)

[02 · Data, tokenization and leakage controls](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/02%20%C2%B7%20Data,%20tokenization%20and%20leakage%20controls%203d88a2d10e44815391dffb1e1533faa8.md)

[03 · Analysis plan and decision rules](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/03%20%C2%B7%20Analysis%20plan%20and%20decision%20rules%203d88a2d10e4481af9a3deb8e29c432c7.md)

[04 · Execution, GPU budget and reproducibility](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/04%20%C2%B7%20Execution,%20GPU%20budget%20and%20reproducibility%203d88a2d10e44811ebd95cc0afce540c9.md)

[05 · Decision sheet, limitations and revision record](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/05%20%C2%B7%20Decision%20sheet,%20limitations%20and%20revision%20reco%203d88a2d10e4481298fd9f24f8a9b7309.md)

[Implementation tasks](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Implementation%20tasks%20facc2f8abdc3499a99aa523d288eaadf.csv)

[Run registry](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Run%20registry%20310a761cff234f6ab5b3564877a7e102.csv)

[Untitled](Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi/Untitled%203d88a2d10e4481aba8eeebca15c90446.csv)