# Original C4 → Python pilot on Kaggle

Prepared 2026-09-29. The user will execute the experiment. No original-corpus scientific results or Kaggle feasibility approval have been produced by this implementation work.

## Start here

1. Save outputs and stop any old, idle GPU session. Its elapsed usage still counts.
2. Upload `artifacts/kaggle-stage1/domain-shift-stage1.zip` as a **private** Kaggle dataset. Attach exactly one version. Kaggle may unpack it automatically; both notebooks accept either extracted files or the ZIP.
3. Import `notebooks/kaggle-stage1-prepare.ipynb`. Set Accelerator **None**, Internet on, and grant the notebook the Kaggle Secret `HF_TOKEN`. Use your approved Hugging Face account. Never paste a token into a cell, report, or chat.
4. Run preparation. It captures source revisions and licenses, prepares original C4 English and `bigcode/the-stack-dedup` Python, audits separation/deduplication, and creates fixed orders. Preserve the complete preparation audit privately.
5. Persist `stage1-online-inputs.zip` as private input for the GPU notebook. Preserve `stage1-reserved-test-SEALED.zip` separately; **never attach the sealed test archive to the pilot GPU notebook**.
6. Import `notebooks/kaggle-stage1-run.ipynb`, attach code and online inputs, select T4 ×2, and use `MODE='preflight'`. Only GPU 0 is used. Enter actual remaining session minutes, remaining weekly GPU quota, and cumulative experiment GPU time outside the runner ledger.
7. Download/persist the full preflight archive. Send the small `stage1-preflight-reports-*.zip` for review. This is the first real-data numerical and feasibility gate.
8. In a fresh session, attach that preflight archive and use `MODE='start'`, with updated usage values and `RUN_PRIMARY=True`. The notebook must reproduce subsequent updates from the persisted checkpoint before admitting seed 101.
9. After each chunk, persist the full `stage1-resume-*.zip` and small report ZIP. For the next session attach only the latest full resume archive, use `MODE='resume'`, and update usage values. **Keep every older full archive locally**: historical weight snapshots are verified and indexed rather than copied into every later archive.

Do not rerun start mode to replace an existing primary run. Resume the same frozen matrix. A reports-only archive cannot resume training. The notebook runs in temporary scratch space; `/kaggle/working` exports also need saving outside the live session. No automatic remote upload is configured. Verify the exported archive is present in a saved output/download before stopping the session.

## Data preparation and interruptions

The prepared WikiText/CodeSearchNet proxy data and old synthetic benchmark cannot satisfy this protocol. Preparation pins actual available source revisions, records repository aliases and license metadata, groups Python repositories before splitting, and separates exact and near duplicates with held-out priority. Five-token shingles, 128 MinHash permutations, 32 × 4 LSH bands, and exact candidate Jaccard ≥0.85 are used. A deterministic sampled missed-pair audit is a check, not a proof that all near duplicates were found.

The raw candidate pool can be much larger than the final arrays, and full-corpus grouping/deduplication can exceed a Kaggle CPU session or memory/disk allocation. Start with CPU preparation; if the host cannot accommodate it, run the same preparation CLI locally with sufficient disk/RAM. Do not substitute data or reduce token quotas to make it fit. Completed shards commit atomically in `candidate-pool.sqlite`; `--resume` requires the same code and source identity. Preserve the entire preparation directory to resume across machines/sessions. Increase `--max-shards` with `--resume` if the bounded pool reports a quota deficit. Partial shards are replayed; no incomplete arrays are admitted.

The online corpus includes 260M web training tokens, 110M Python training tokens, and 2,097,152 development labels per domain. Quick development uses the first 262,144 labels of the full development set. Reserved test has 8,388,608 labels per domain and remains sealed. Orders are paired across conditions, without repetition within a trajectory. Source files and arrays are hash-bound to the freeze.

## Recorded T4 amendment

`configs/kaggle-stage1.json` records the user's authorized FP16 amendment to the original BF16 protocol. It uses FP32 master weights, FP32 normalization/calibration statistics and loss calculation, FP16 autocast and gradient scaling. Forward nonfinites fail validation; gradient scaling does not repair forward overflow. Do not pool FP16 results with a later BF16 replication.

The model, original datasets, seeds 101/102/103, token budgets, effective batch, primary estimands, and scientific thresholds remain unchanged. The primary allocation is RMS and Taper-minus; optional Taper-plus is omitted before admission. Execution is eager, single GPU, microbatch 2 × accumulation 16 = 32 sequences of 512 labels. The second allocated T4 is idle. No distributed or TPU implementation is admitted by this runner.

Preflight uses engineering seed 997. It measures RMS, genuinely calibrated Taper, intermediate-gate and zero-gate execution, development evaluation, diagnostics, and checkpoint I/O. Repositioned taper-clock timing probes are explicitly engineering fixtures, not evidence of long-run stability. A new engineering admission tolerance requires quick-development CE to differ from FP32 by at most **0.005 nats/token** at the tested states. This tolerance is fixed before primary runs and is not an H1 effect threshold. Real-array validation, synthetic acceptance tests, finite numerics, resource projection, and external-session restore must all pass.

Source mapping: Taper's interpolation follows equation 3, calibration follows equation 4, with EMA freezing and gain copying at calibration end. Final RMSNorm remains active. This is the project's architecture adaptation, not an exact reproduction of the paper's architecture/training setup. Source capture retains the paper and actual nanoGPT commit/model/license for audit. Training diagnostics use the last 256 training windows per domain; these can overlap trained documents and are labeled accordingly. Relative prefix gap uses the RMS CE denominator.

## What numbers determine continue or stop?

The original endpoint is continuation update **6,104**, after each condition's 9,156-update web prefix. Both web and Python children restore that condition's identical complete prefix state. The primary allocation processes **2,100,166,656 supervised tokens**. Decisions require all three paired seeds, both conditions, both branches, scheduled evaluations/diagnostics and completion receipts.

Define `D = (Taper Python-branch web CE − Taper web-branch web CE) − (RMS Python-branch web CE − RMS web-branch web CE)`. Positive D means **more excess forgetting for Taper**, supporting the normalization-removal hypothesis; it does not mean Taper is a better language model. All CE differences below are nats/token.

The machine-readable thresholds are in `configs/kaggle-stage1.json`; `analysis/decision.py` implements the ordered rules and rejects changes to the original scientific thresholds:

- **Invalid/incomplete:** missing runs/evaluations, unresolved data or correctness failures, nonfinite primary training, or resource termination. This cannot justify a scientific kill decision.
- **Comparability/adaptation limited:** any seed's absolute relative prefix web-CE gap exceeds 2%, or either condition improves Python non-whitespace CE by less than 0.05. Interpretability failed.
- **Opposite direction:** mean endpoint D ≤ −0.03, after the validity/comparability gates.
- **Proceed to design the next study:** mean endpoint D ≥ 0.03; D > 0 in every seed; mean D at update 3,050 ≥ 0.015; mean Q > half mean D; and mean matched-adaptation forgetting difference > 0 at the furthest common eligible checkpoint among 1,525/3,050/6,104. Here Q removes the web-control between-condition gap change, isolating direct excess forgetting on the Python branch.
- **Stop—small observed effect:** mean endpoint D < 0.015 and no early mean quick-development D ≥ 0.05 through update 1,000. This is a pilot screen, not evidence of equivalence.
- **Transient only:** endpoint D < 0.015 but that early transient threshold was reached.
- **Inconclusive:** valid/comparable evidence that meets none of the above outcomes. Values between 0.015 and 0.03 do not automatically justify continuing.

For practical planning, only the complete “proceed” category earns a next-study recommendation. The other valid categories do not authorize more runs or an enlarged budget automatically. Incomplete or adaptation-limited results diagnose feasibility/design limitations rather than establish the hypothesis is false. Three seeds do not support a broad confirmatory claim.

## Time, quota, and the live session

On 2026-09-30 the user authorized a **50 cumulative active GPU-hour cap on free Kaggle**, spread across sessions and weekly resets, replacing the paid-hardware 20 optimizer / 24 total-hour restriction. The optimizer guard is also 50 hours and remains inside the same total; it is not an additional allowance. Training still stops at the original fixed token/update endpoints, never at a longer outcome-selected endpoint. The 15% feasibility margin is retained. Preflight includes all prior experiment GPU usage, failures, evaluation, diagnostics and checkpoint overhead. The original budget policy is preserved in `configs/kaggle-stage1-24h-baseline.json`.

The user's first real-data T4 preflight passed numerical/replay validation and projected 19.01 optimizer hours plus 13.13 overhead hours, or 37.31 hours including its reported prior usage and margin. That estimate fits the amended cap, subject to complete cumulative prior usage. Since budget/config/code identity is frozen into admission evidence, run one fresh preflight with the amended bundle; retain the old report as historical evidence. CPU data preparation need not be repeated. A new-session restore must still pass before seed 101.

Weekly quota renewal changes only `WEEKLY_GPU_HOURS_REMAINING`; it never resets the experiment ledger or `EXTERNAL_GPU_HOURS`. Save and verify the complete resume archive before a session ends. In the next session/week attach the latest archive and use `MODE='resume'`, with the same code, online data and compatible recorded runtime. A runtime mismatch stops execution for investigation. Checkpoint pauses preserve the intended FP16 trajectory subject to the tested replay tolerance; they do not establish bitwise equivalence to BF16 or different GPU hardware.

Kaggle documents a 12-hour CPU/GPU session limit; use the actual UI timer and quota, which can differ from nominal limits. The notebook takes the smaller of current session time and remaining weekly quota. Default work chunks are 110 minutes with a 10-minute save reserve, shortened when necessary. Stop and save when requested; do not rely on the platform killing a process cleanly.

`EXTERNAL_GPU_HOURS` is cumulative experiment GPU time **outside** the runner ledger: old validation/smoke attempts, preflight, setup, idle time, restore checks, and exports. Include all such time, never decrease it on resume, and avoid double-counting training chunks already charged in `ledger.json`. Check the platform usage against the ledger. Newly elapsed setup/export time must be included at the next invocation. The ledger reserves a full chunk before work and reconciles clean finishes; an unclean attempt keeps its conservative reservation. Use `RECOVER_UNCLEAN=True` only after confirming the prior worker has stopped. Run only one worker.

## Checkpoint contract

Full checkpoints preserve model/calibration/gates, Adam moments and counters, scaler, CPU/CUDA and sampler RNG, fixed-order position, branch identity, update clock, and provenance. Atomic generations have checksums and retain the previous generation for investigated recovery. Corruption fails closed; it does not silently roll training backward. Full checkpoints occur at most 15 minutes apart and before evaluation; a deadline or interrupt requests a stop at an update boundary. A hard host kill can lose work since the last externally persisted artifact.

At a completed prefix the runner deliberately pauses until its switch checkpoint has been exported, persisted, and imported. Only then may its two children start. Full-development weight snapshots and sufficient statistics are saved; switch and final states remain complete. Archives validate hashes and CRC and omit redundant checkpoint aliases. Both retained ZIPs and Kaggle's auto-extracted archive directories are supported.

Keep all historical full archives and the latest small report. Do not edit checkpoint files, freeze values, orders or source code mid-matrix. Code/runtime changes fail provenance checks and require investigation. Do not delete old archives merely because a newer resume ZIP exists. Check available output storage before saving; the notebook exports one chunk per invocation and does not treat temporary scratch as a backup.

## PyCharm and Kaggle

PyCharm Pro supports external Jupyter servers: **Tools → Add Jupyter Connection → External Server**, enter the server URL/token and test the connection. If your Kaggle session exposes an external-editor/Jupyter connection URL, this is the route to try. Authentication compatibility with your Kaggle account has not been tested here. This connects notebook execution; it is not a general SSH remote Python interpreter. Session lifetimes and GPU quota still apply.

The reliable file-based workflow is editing locally in PyCharm, uploading the prepared code/notebook, and using the official Kaggle CLI (`kaggle kernels push`, `status`, `pull`, `output`) from its terminal. Never include HF credentials in uploaded code.

References: [Kaggle notebook limits](https://www.kaggle.com/docs/notebooks), [efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), [Kaggle external editor information](https://www.kaggle.com/notebooks/welcome), [PyCharm Jupyter connections](https://www.jetbrains.com/help/pycharm/configuring-jupyter-notebook.html), [Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md), [Taper paper](https://arxiv.org/html/2602.10408v1).

## Evidence status

Local synthetic tests verify decision rules, missingness, data separation, session deadlines, interrupted-budget accounting, archive integrity, and actual CUDA FP16 checkpoint replay across calibration activation. Local tests alone do not establish real-data T4 feasibility; the user has since supplied a passing real-data numerical preflight and the runtime projection described above. The two notebooks intentionally require user-entered usage and an explicit primary-run setting. The old synthetic T4 benchmark measured roughly 44.7k RMS and 46.4k zero-fixture Taper tokens/second over ten timed updates; those numbers cannot replace the new real-data admission gate.
