# Sustained training on the RTX 2060

The original local lab is a **smoke-test/teaching fixture**: 128 generated training
documents and 24 development documents per domain, a two-layer byte model, context
64, and only 300 updates by default (six with `--smoke`). Repeating those templates
for hours mostly exercises memorization. The larger lab below trains the existing
17.7M-parameter, six-layer RMS model from scratch on real text, with actual GPT-2
token IDs and all 50,257 output classes in use.

## What uses the GPU?

Dataset size controls how much different material is available. **Microbatch size,
context length, model computation and precision** determine how much work the GPU
does at a time. A dataset can be much larger than VRAM: the loader reads compact
token arrays from disk and transfers only the current batch.

Gradient accumulation combines several microbatches before changing the weights.
It increases the effective batch without holding every microbatch's activations
in memory at once. It does not, by itself, fill an underused GPU.

The short benchmark on this laptop selected:

- **Context:** 512 GPT-2 tokens.
- **Microbatch:** 8 sequences; **accumulation:** 2; **effective batch:** 8,192 predicted tokens.
- **Precision:** FP16 mixed precision, with FP32 master weights, dynamic loss scaling,
  and CUDA fused AdamW. BF16 is not required.
- **Measured:** about 50,800 training tokens/second, 97% median device utilization,
  and 3,608 MiB peak PyTorch reserved memory for this candidate.
- The benchmark's maximum observed temperature for this candidate was 71°C.

These are short measurements, not an hours-long thermal test or a guarantee.
Reserved memory excludes other applications and some driver memory. The benchmark
leaves memory headroom and chooses the fastest measured feasible batch; consuming
the last available megabyte is not the objective. Measurements are recorded in
`artifacts/local-lab/rtx2060-benchmark/benchmark.json`.

## Datasets

**Prose:** [Salesforce WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext),
assembled into Wikipedia articles before tokenization. It is a substantial prose
corpus, but it is not a general-web corpus such as C4.

**Python:** [CodeSearchNet's Python subset](https://huggingface.co/datasets/code-search-net/code_search_net),
using complete function strings, including their docstrings. These are functions
from repositories, not complete projects. The [upstream project](https://github.com/github/CodeSearchNet)
documents the source-repository licensing information; source licenses vary.

Preparation pins the source revisions, saves upstream cards and file checksums,
and retains per-document source identities. It uses only upstream **train and
validation** files. It never downloads upstream test files or reads the pilot's
reserved test directory. Validation is treated as development data in this lab.

Defaults retain **up to 50 million training tokens per domain** and 131,073
development tokens per domain. The cap is deterministic: it takes a prefix of
usable source documents rather than pretending to be a random representative
sample. The manifest reports actual counts if a source is exhausted. The final
selected document can be truncated at the token cap, which is recorded.

Exact duplicate documents are removed with development precedence across both
domains. Training excludes any selected development article title (prose) or
repository (Python). There is no repository-alias reconciliation or near-duplicate
audit. These protections are useful for learning but do not meet the full pilot's
data acceptance requirements.

The loader packs documents with GPT-2 EOS boundaries. Packed sequences may cross
document boundaries within a split; they never cross a train/development split.
Training samples fixed windows with replacement. Token counts divided by corpus
size describe average exposure, not guaranteed complete epochs.

## 1. Install the small additional data dependencies

Run these commands in PowerShell from the project directory. CUDA PyTorch is
already installed in `.venv` on this laptop.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[local-data]'
```

## 2. Download and prepare the larger corpora

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_corpus
```

This has **not been launched as part of the setup-only request**. Expect roughly
**710 MB of corpus downloads** if the first WikiText training shard supplies the
cap; another 157 MB is fetched if the second is needed. About 200 MB of training
token arrays are written for the default caps, plus provenance and development
data. Allow several GB for source caches, outputs and checkpoints.

Source files are cached under `data/local-downloads`; the prepared corpus is
under `data/local-large`. Network downloads resume where supported and verify
the published source hashes. A failed preparation can be rerun with the same
settings; a completed corpus is never silently replaced.

For a different cap, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_corpus --train-tokens 100000000 --output data/local-100m
```

The first tokenizer use also downloads GPT-2 encoding files. No pretrained model
weights are downloaded and no downloaded Python source is executed.

## 3. Optional: repeat the benchmark for your current laptop conditions

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_training benchmark --output artifacts/local-lab/benchmark-new --hours 2
```

This runs short hardware checks with synthetic token IDs, **not** a multi-hour
experiment. It saves `recommended.json`, including a step count estimated for
the requested total duration. For a four-hour estimate, change `--hours 2` to
`--hours 4`. Use a fresh output folder. Longer measurement windows can be requested
with `--seconds 60` per candidate to observe more of the warm-up behavior.

The search compares microbatches 1, 2, 4, 8 and, when projected memory permits, 16.
Every candidate uses the same 8,192-token effective batch. A larger candidate is
skipped when its projected memory exceeds the headroom budget; this is not an
exhaustive search of every possible GPU kernel or model configuration.

## 4. Launch a sustained experiment when you are ready

The checked-in profile uses the measured batch/precision and approximately two
hours of total work, **across all three stages**, not two hours per stage:

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_training train --config configs/local-2060.json --data-dir data/local-large --output artifacts/local-lab/large-run-01
```

The profile runs 19,416 prose-prefix updates, then 9,708 continued-prose updates,
then 9,708 Python updates: **318,111,744 predicted tokens** in total. Each branch
restores the same prefix weights, optimizer moments, loss scaler and random state.
The global warmup/cosine learning-rate clock continues from the prefix in both
branches; it does not restart at the domain switch.

The duration is a throughput estimate with an overhead allowance, not a deadline.
Power mode, other GPU applications, thermals, disk writes and evaluation affect it.
Use the laptop's normal plugged-in performance mode and watch sustained throughput.
This setup does not change GPU clocks, power limits or fan settings.

For a newly benchmarked profile, replace the config path with that benchmark's
`recommended.json`. Do not accidentally include `--smoke`: this is a separate runner.

## 5. Monitor, stop and resume

In another terminal:

```powershell
nvidia-smi -l 1
Get-Content artifacts/local-lab/large-run-01/status.json
Get-Content artifacts/local-lab/large-run-01/metrics.csv -Tail 5
```

The saved profile evaluates 128 fixed, evenly spaced development windows per
domain every 500 updates, and saves a completed checkpoint at those boundaries.
This is **quick development evaluation**, not full-corpus evaluation. CE is
token-weighted within those fixed windows; lower is better. Compare prose CE
before and after the switch, and against the continued-prose branch. Training
loss is the most recent effective update's loss, not a smoothed interval average.

For a bounded trial after data preparation, add `--max-updates 20` to the train
command. It saves at a completed update and pauses without changing the planned
learning-rate schedule or total experiment. You can also press Ctrl+C. An
interruption resumes from the latest completed checkpoint, so updates after that
checkpoint may be repeated. A half-finished update is never saved.

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_training resume --output artifacts/local-lab/large-run-01
```

Keep the entire run directory, especially `latest.pt`, `switch.pt` and the
manifest. Resume validates configuration, token files, runner/model code and
PyTorch version; changing them requires a new run. Load only trusted local
checkpoints. Checkpoints preserve the FP16 scaler and RNG in addition to weights
and Adam state. Automatic overflow retries repeat the same microbatches at a
smaller scale and do not advance the logical step.

`status.json` shows progress; `metrics.csv` records development loss, gradient
norm, LR, throughput, peak allocated memory and overflow retries. Resume rolls
metrics back to the saved checkpoint's history. `summary.json` appears only
after all three stages complete. It reports prose deterioration and the excess
prose loss relative to continued prose training.

## Relation to the research

This is a more informative local baseline and an opportunity to discover loader,
memory, stability, overfitting and recovery problems. It still tests RMS only,
uses substitute corpora and an exploratory schedule, and does not establish the
RMS-versus-Taper H1 endpoint or change protocol v3. Increasing GPU utilization
does not replace the scientific controls, multiple seeds and data-validation work.
