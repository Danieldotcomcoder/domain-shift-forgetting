# Two-T4 original H1 experiment

Prepared 2026-10-01. The user authorized two independent workers and a **50 elapsed notebook-hour** cap. Local synthetic/CUDA tests are distinct from the required real two-T4 Kaggle preflight. No scientific primary runs were executed by this implementation work.

## Files and first run

Use `artifacts/kaggle-paired/domain-shift-paired.zip` and `notebooks/kaggle-stage1-paired.ipynb`. Upload the code ZIP privately, and attach your existing `stage1-online-inputs` dataset. The original prepared corpus and orders are reused; CPU preparation need not be repeated. Keep the sealed test archive separate.

Set Kaggle Accelerator to T4 ×2, enable Internet for dependencies, and start with `MODE='preflight'`, `RUN_PRIMARY=False`. Enter current remaining session minutes, remaining weekly GPU quota hours, and `EXTERNAL_NOTEBOOK_HOURS`. The coordinator has CUDA visibility disabled and launches each worker with exactly its own GPU visible. Do not change those environment assignments to `0,1`.

Preflight runs RMS on physical GPU 0 and Taper-minus on physical GPU 1, using engineering seed 997. Both run acceptance tests, then synchronize the start of calibration, intermediate and zero-gate timing phases. Taper calibration is genuine; later clock repositioning is a timing fixture, not long-run evidence. Full/quick evaluation, diagnostics and checkpoint I/O are measured under concurrent execution. FP32-reference checks are measured for correctness but are not charged as recurring quick evaluation in the projection. Each worker writes its own complete checkpoint and expected five-update continuation, and tests reconstruction into fresh objects.

The forecast includes the slower worker's projected completion time, cumulative external usage, 30 minutes of coordination/import/export allowance, and a 15% margin. It reports each worker's projected cost and their sum separately. A forecast is an admission estimate, not a speedup guarantee. If either worker fails validation or the budget projection fails, do not start primary training. Save the small paired-preflight-reports ZIP for diagnosis.

On success, save the **full paired-preflight ZIP**, then stop that session. In a fresh session, attach exactly that full archive, the same code bundle, and online data. Set `MODE='start'`, `RUN_PRIMARY=True`. Both workers must reproduce their saved continuations from the externally persisted archive before a shared scientific freeze is written. The following cell begins the first bounded training chunk.

Old single-GPU preflight archives do not admit this implementation. Old single-GPU primary checkpoints cannot be silently converted into paired runs. If primary training has already started, preserve that run and resolve the execution transition explicitly before starting another matrix. Starting twice creates a second experiment rather than resuming the first; use resume after the first admitted chunk.

## Worker isolation and paired science

RMS is fixed to GPU 0, Taper-minus to GPU 1; each processes seeds 101, 102, 103 in order. Workers may have different current update counts. They never share optimizer or RNG state, and there is no distributed gradient synchronization. Each retains effective batch 32, context 512, FP16 with GradScaler and FP32 normalization/calibration statistics, canonical paired initialization, identical per-seed data orders, and the original prefix and continuation update budgets. Model, loss, optimizer-update and scientific decision formulas are unchanged.

Each worker's condition-specific prefix independently branches into web and Python by restoring its exact full switch state. The prefix must be exported and verified on import before either branch starts. If one worker reaches that boundary first, the coordinator requests a safe stop from its peer. This can produce unequal saved positions and additional short chunks; neither position is discarded.

Paths are isolated under `workers/RMS` and `workers/Taper-minus`. There is one shared freeze whose hash binds each worker's checkpoints and observations. The parent report reads both worker namespaces and still requires all three paired seeds, both conditions, both branches, all scheduled measurements and completion receipts. An incomplete matrix remains incomplete, never a negative finding. The original threshold configuration is preserved.

## Each subsequent session

1. Save both the full `paired-resume-*.zip` and small `paired-reports-*.zip` after every chunk. Confirm the full archive is available outside the live session before stopping it.
2. Keep all historical full archives: older full-development weight snapshots remain in the archive that first persisted them. New archives carry their provenance indexes rather than duplicating all weight history.
3. In the next session attach only the latest full paired resume archive, plus the same code and online data. Set `MODE='resume'`, `RUN_PRIMARY=True`, and update the current time/quota values.
4. Both retained ZIPs and Kaggle-auto-extracted archive directories are supported. Import verifies hashes and projects global archive indexes into each worker's namespace.
5. The coordinator stops both before exporting. One combined archive contains both workers' committed states, logs and receipts. Do not export a worker directory separately while training is active.

Chunks default to 110 work minutes and a 10-minute save reserve, shortened to fit the current session/quota/budget. The coordinator holds an OS lock to prevent a second live coordinator using that root. A failure or interrupt requests both workers to stop; a nonresponsive process is eventually terminated and the attempt is recorded as failed. The notebook interrupt handler gives the coordinator time to stop both workers. Full export refuses an active-writer marker or a still-running coordinator ledger; this also prevents an interrupted notebook from archiving files while workers might still be writing. A hard platform kill can lose updates after the latest externally persisted archive. Local checkpoints alone are not durable across session loss.

For an unclean attempt, `RECOVER_UNCLEAN=True` is allowed only after confirming the prior workers/session have ended. The prior reserved time is conservatively retained. This setting does not bypass failure markers, corrupt files, changed data/code/runtime, or missing validation. Report any failure for investigation; do not delete its evidence or replace its seed.

## Hours: one cap, two measurements

The user selected **50 cumulative elapsed notebook hours**, across weekly quota resets. Two workers running concurrently for one hour spend one elapsed notebook hour and roughly two summed worker hours. `notebook-ledger.json` is the authoritative experiment time ledger; worker `session-ledger.json` files and coordinator lifetime records provide separate per-worker accounting. Worker lifetime includes its own setup/validation/wait time while the process is alive; it is not a measurement of kernel utilization.

`EXTERNAL_NOTEBOOK_HOURS` means cumulative experiment GPU-session time outside the parent coordinator ledger. Include previous single-GPU smoke tests, all preflights (including failures), and setup/idle/import/export/recovery time. Count overlap between the two GPUs once. Exclude CPU-only preparation, stopped-session time, unrelated projects and coordinator-run chunks already in the ledger.

The notebook automatically adds elapsed setup/import/restore time since its settings cell to the external amount just before a training chunk, and prints that new external value. For the next invocation start from that printed value, then add exports and later idle time. Do not count automatically added setup twice. The preflight summary's `prior_notebook_hours` includes its supplied external amount plus that preflight invocation; add subsequent outside time before `MODE='start'`.

A weekly quota reset changes `WEEKLY_GPU_HOURS_REMAINING` only. It does not reset the experiment ledger or external total. A 50-hour cap does not add training tokens: the run always finishes at its original fixed endpoints. The current single-GPU historical forecast is not reused as a claimed two-GPU measurement.

The parent ledger reserves the whole chunk before launching workers. It records actual elapsed time for clean completion and separately records the sum of worker lifetimes; a killed coordinator retains its conservative reservation. The legacy `external_gpu_hours` key inside the underlying ledger is explicitly labeled `budget_unit: elapsed_notebook_hours` for this paired implementation.

## Results and remaining validation

Send the small paired reports ZIP after each chunk. Positive primary D means more excess web forgetting in Taper under the Python shift; it is not a claim that Taper is a better language model. All scientific rules remain those in the original runbook and `configs/kaggle-stage1-paired.json`.

Checkpoint pauses preserve the intended FP16 experiment subject to the tested replay tolerance. Exact equality across hardware or BF16 is not promised. The real T4 ×2 concurrency benchmark and fresh-session dual restore are required on Kaggle; local tests cannot substitute for that hardware evidence.
