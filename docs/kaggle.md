# Kaggle free GPU testing

This is an engineering benchmark, separate from the frozen H1 pilot. It does not
change the three seeds, token budgets, precision declaration, or 24 rented GPU-hour
cap. Record a revised compute plan before scientific runs; notebook quota-hours
and summed GPU-hours are different units when two GPUs are used.

Kaggle's [GPU usage documentation](https://www.kaggle.com/docs/efficient-gpu-usage)
describes a weekly 30-hour quota, sometimes higher with availability. Check the
actual account quota, accelerator and session deadline in the UI. Hardware options
and session limits can change. This runner uses one GPU, FP16 with dynamic gradient
scaling, FP32 master parameters and FP32 norm statistics. It does not assume BF16.

## Small benchmark: no corpus or credentials needed

1. Run `python scripts/package-kaggle.py` from this repository.
2. Upload `artifacts/kaggle/domain-shift-kaggle.zip` to a **private Kaggle Dataset**.
3. Import `notebooks/kaggle-benchmark.ipynb` into a Kaggle Notebook and attach that
   Dataset. The setup accepts either the ZIP or the extracted `src` directory that
   Kaggle may expose. Enable an available GPU accelerator. Internet is not required.
4. Run the notebook. It selects GPU 0 and runs four cases serially: RMS and
   zero-gate Taper-minus, each in FP32 and FP16. Defaults are context 512,
   microbatch 2, accumulation 16: **16,384 labels per update**. If memory runs out,
   use microbatch 1; accumulation changes to preserve the effective batch.
5. Save a notebook version with outputs, download the printed `kaggle-results-*.zip` archive, and verify
   it is accessible after the session ends. Send back the archive or its four
   `benchmark.json` files and your available quota/accelerator details.

Each case has two warmup updates, ten timed updates, evaluation, checkpoint I/O,
and a one-update restore/replay comparison of model, Adam, scaler and sampler state.
Every case uses the same initialization seed and synthetic token stream. Both
precision CEs are measured on identical initial and final model states. Reports
include throughput, peak reserved memory, overflow retries, evaluation/save/restore
times, and replay success. `passed` means execution and replay passed; it does not
mean the precision CE difference is scientifically acceptable. Inspect that
difference and the full FP16/FP32 trajectories before planning longer tests.

Budget roughly 10–20 minutes for a first attempt; this is a planning allowance,
not a measured T4 runtime. Stop after the four cases. No large download or full
experiment starts automatically. On failure the notebook still creates the results
archive, including the failure report, then raises the failure.

The zero-gate case deliberately sets `c=1` and the clock to 6,104 on a fresh model.
It exercises the actual zero-gate operator but **does not simulate 763 calibration
updates or a trained taper trajectory**. Before H1 runs, repeat precision/stability
checks on a genuinely calibrated checkpoint after the gate reaches zero and run
the full scientific acceptance checks. A one-step replay is also not a long soak.

## Checkpoint durability and moving sessions

The exploratory runner now writes full state into an immutable generation, fsyncs
it, then atomically publishes a checksummed `.pt.json` pointer. It retains the
previous committed generation. Restore checks the hash, schema, code/data/config
identity, runtime and CUDA topology before changing live state. State includes
weights, Adam moments/learning rate, scaler, update/stage counters, sampler, Python,
NumPy and Torch RNGs. The LR schedule is a pure function of config and the saved
update clock. Saves reject unfinished calibration state. Failed updates are never
saved. The `.pt` file is a convenience alias; restore uses the pointer.

**Copy/archive the whole output directory, including hidden files**, not just
`latest.pt`. `/kaggle/working` is session storage, not an external backup. An atomic
write cannot protect against losing the session's entire disk. Save/download
outputs at each planned pause, before the platform deadline; verify the archive
exists outside the running session. Automatic remote upload is not configured.
If corruption is reported, fail closed and recover the previous generation by
explicitly replacing the pointer's `file`/`sha256` with its `previous` record on a
copy of the run. Never bypass hash verification. Files are trusted local artifacts.

For optional real-data RMS testing, separately attach the prepared `local-large`
corpus described in `docs/local-large.md`. Use the existing local-training CLI with
`--max-updates 10` and an output under `/kaggle/working`. To resume next session,
extract the entire saved run there and use:

```bash
python -m domain_shift_forgetting.local_training resume \
  --output /kaggle/working/my-run --data-dir /kaggle/input/YOUR-DATASET/local-large \
  --max-updates 10
```

Keep the exact code bundle and PyTorch/runtime/device allocation across sessions.
Changed identities are rejected. Older pre-integrity checkpoints remain readable
with the original runner; this version intentionally does not silently resume them.
Kaggle may update its base environment, so capture the notebook's `pip freeze` and
hardware report. After one-GPU validation, separate processes may use GPU 0 and 1
with `CUDA_VISIBLE_DEVICES`, distinct output directories and the same settings.
Benchmark concurrent operation separately; do not assume double throughput.

If setup was already run, rerunning it creates a fresh results directory. Use the
new `results` variable in the benchmark cell; the archive is named after that
directory. The standalone replacement setup cell is `scripts/kaggle-setup-cell.py`.
