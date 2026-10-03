# Laptop learning lab

**For real datasets and hour-scale GPU training, use [the larger local setup](local-large.md).**
This page documents the original small smoke-test/teaching lab.

This is an executable learning sandbox for a 6 GB RTX 2060. It teaches next-token
prediction, optimization, evaluation, and domain-shift comparisons. It does not
execute protocol v3 or establish whether TaperNorm increases forgetting.

## Start here (PowerShell, from the project folder)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_lab --device cuda --smoke
```

The setup creates a project-local environment and installs CUDA-enabled PyTorch,
NumPy, pytest, and this package. The initial PyTorch download is about 2.6 GB;
allow several GB of disk space for installation and caching. No corpus or model
weights are downloaded. You do not need to activate the environment.
The explicit PyTorch build follows the [official versioned installation commands](https://pytorch.org/get-started/previous-versions/).
The script is safe to rerun after an interrupted download.

The smoke run performs two prose-prefix updates, saves a checkpoint, then performs
two updates in each continuation branch. It verifies plumbing; two steps cannot
demonstrate meaningful learning. Use `--device cpu` if CUDA is unavailable.
With the default `--device auto`, the manifest records the actual selected device.

## First learning run

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_lab --device cuda
```

Defaults: two transformer layers, width 256, four attention heads, context 64,
microbatch 2, four accumulated microbatches, 100 prefix updates and 100 updates
per continuation branch. This is training a small language model from random
weights, not fine-tuning an instruction model. FP32 keeps the first run simple
and avoids depending on BF16 support. No mixed precision, compilation, or data
loader workers are required.

Each effective update predicts **2 × 64 × 4 = 512 tokens**. A token here is one
UTF-8 byte, with token 256 marking a document end. It is not a GPT-2 token.
The model sees a sequence and predicts the next token at every position; loss
backpropagation produces gradients, AdamW changes the weights, and evaluation
checks predictions on documents that were not used for training.

The bundled original fixtures resemble prose and Python. Exact documents are
disjoint across train/dev and domains, but their templates overlap heavily.
They are deliberately easy pipeline fixtures, not a representative web/code
benchmark. Repeated sampling and memorization are expected. Windows stay inside
each document, use shifted next-token labels, and discard short tails.

The run first trains on prose. It then restores the same prefix model, AdamW
state, and random state for each of two branches: more prose versus Python.
The learning rate stays constant; the optimizer does not restart at the switch.
The branches run sequentially, so only one model occupies GPU memory.

## Read your run

Every run creates a fresh folder in `artifacts/local-lab/`:

- `manifest.json`: settings, software/device, parameter count, data hashes.
- `metrics.csv`: initial, prefix, and continuation evaluations, flushed as they run.
- `summary.json`: completion status and the endpoint comparisons.
- `switch.pt`: complete local state at the prose/Python fork.
- `web-final.pt`, `python-final.pt`: each branch's final state.
- `failure.json`: error details if the run fails after creating the output folder.

**CE (cross-entropy)** is average next-token prediction loss in natural-log units;
lower is better. **Perplexity** is `exp(CE)`; lower is also better. These byte-level
values cannot be compared directly with GPT-2-token research metrics.

Watch for falling training CE and improving held-out CE. Training CE falling
while development CE rises suggests overfitting or a domain mismatch.
`train_ce` is the last update's mean, not the mean since the previous evaluation.
Development CE covers every retained window in that split, weighted by labels.

`web_deterioration_after_python` is web CE after Python training minus web CE at
the switch. Positive means web prediction got worse. `excess_web_ce_after_switch`
subtracts the continued-prose branch endpoint instead; it accounts for what
continued prose training would have done. This is a single-model teaching
comparison, not the RMS-versus-Taper difference-in-differences endpoint.

`grad_norm` is measured before clipping; `clipped` means the norm exceeded 1.
Occasional clipping is normal. Nonfinite losses/gradients stop the run.
Training tokens/second covers the last training update, with CUDA synchronized;
it excludes development evaluation and checkpoints. Elapsed time includes both.
Peak allocated/reserved MiB are PyTorch memory measures, not total device use.

## Experiments to learn from

Change one setting at a time and retain the same seed when comparing settings.

1. Increase steps to watch learning and later overfitting:

   ```powershell
   .\.venv\Scripts\python.exe -m domain_shift_forgetting.local_lab --device cuda --prefix-steps 500 --continuation-steps 200
   ```

2. Reduce memory use with `--batch-size 1 --context 32`. Increase accumulation
   to preserve the effective token batch if that is the intended comparison.
   Context changes also change retained data windows and the prediction task.
3. Try `--lr 0.0002` and compare loss and gradient behavior.
4. Supply real, small, locally held text documents using `--data-dir PATH`.
   Put separate UTF-8 `.txt` files under `web/train`, `web/dev`, `python/train`,
   and `python/dev`. Split by original source document/repository before copying.
   Use only data you may use; do not point this at reserved research test data.
   Exact duplicate documents are rejected; near-duplicate/source-group detection
   is not implemented. Keep the collection small: this loader materializes it in RAM.

## Exercise the existing model

```powershell
.\.venv\Scripts\python.exe -m domain_shift_forgetting.local_lab --device cuda --model pilot-rms --smoke --batch-size 1 --accumulation 1
```

This uses the existing six-layer, 17.7M-parameter RMS transformer, including its
50,257-way output, with short sequences and the local optimizer loop. Input IDs
still mean bytes in this lab; unused output classes remain in the softmax. It is
an architecture/gradient/memory rehearsal, not GPT-2-tokenized research training.
Do not compare its raw loss to the tiny model's loss. This option exercises more
of the existing model code but does not validate the protocol training loop,
Taper calibration/gating, BF16, full context/batch, real data, or rented-GPU throughput.

The scientific architecture, schedules, configuration, freeze gates, and reserved
test data are unchanged. Runtime validation did expose a small arithmetic mismatch:
Taper's normalized path now uses the same reciprocal-square-root calculation as
RMSNorm, preserving its formula while matching the RMS endpoint's FP32 rounding.
A successful lab run does not make T01–T20 complete.

## Failures and checkpoint scope

- CUDA unavailable: check the setup output and use the project's Python executable.
- Out of memory: close GPU-heavy applications, use the tiny model, reduce batch/context,
  and retry into a fresh output directory. An existing output folder is never reused.
- Slow laptop runs: plug into power and compare throughput across short runs;
  temperature/power limits can change speed. Ctrl+C stops and records an interruption.
- Data error: all four folders need at least one document longer than the context.

The runner restores the switch checkpoint automatically before both branches.
Tests verify that restoring model, optimizer, and RNG reproduces the next CPU
update exactly. There is no arbitrary interrupted-run resume CLI yet: rerun short
experiments into a fresh directory. Final checkpoints are inspection/replay assets;
only load your own trusted checkpoints. Local checkpoints are not protocol checkpoints.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_lab.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The first suite checks data isolation, causal prediction, token-weighted evaluation,
checkpoint replay, matched branch starts, output collision protection, and invalid
settings. The broader suite checks the existing prepared primitives; passing it
does not satisfy the full scientific acceptance gates.

For installation followed by all tests and both short CUDA smoke runs, use
`powershell -ExecutionPolicy Bypass -File .\scripts\validate-local.ps1`.
It writes `status.json`, `validation.log`, package versions, and the two runs
inside a fresh `artifacts/local-lab/validation-*` directory. A status of
`waiting_for_download`, `installing`, or `testing` is not a pass; only `completed`
means every command succeeded. A failed check stops the sequence.
