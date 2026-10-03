# 04 · Execution, GPU budget and reproducibility

[Domain-Shift Forgetting · Stage 1 — H1 Research Pilot](../Domain-Shift%20Forgetting%20%C2%B7%20Stage%201%20%E2%80%94%20H1%20Research%20Pi%203d88a2d10e448149ac9df736861f0d40.md)

## Resource policy

Hard cap: 24 actual rented GPU-hours including failed attempts, setup, compilation, calibration, evaluation, checkpoint I/O and training. CPU preprocessing happens before rental. No GPU has been rented by creating this project.

Plan at most 20 hours of optimizer-update workload and reserve four for all other GPU-billed activity. These are allocations, not verified runtime. Benchmark phase-specific training times and separately measure evaluation/I/O overhead; do not count the same evaluation overhead both in throughput and the four-hour reserve.

Projection: sum(update-count-phase × measured-seconds-per-update-phase) + scheduled-evaluation-seconds + startup/checkpoint/diagnostic-seconds. Use an additional 15% uncertainty margin on this projected total; admit the allocation only if it still fits 24 hours.

Mandatory primary: 2,100,166,656 tokens, requiring about 29,200 training tokens/s for 20 training hours. Full three-condition design: 3,150,249,984 tokens, requiring about 43,750. These are requirements, not GPU performance predictions.

## Rental choice

Default: one on-demand RTX 4090 24GB; at least 32GB host RAM, six CPU cores/vCPUs where available, and 100GB working storage including raw preparation remnants and checkpoint growth. The 100GB storage allowance replaces v2's tight 50GB estimate; measure actual use and keep durable copies of prefix states.

Initial rental block: up to two hours for correctness and benchmark on prepared data; this is part of the cap, not additional. Extend hourly only after the feasibility gate.

Rates observed 11 September 2026: Runpod Pod RTX 4090 USD 0.74/hour; RTX 5090 0.99/hour; RTX A6000 0.53/hour; A100 PCIe 1.59/hour. Recheck checkout and availability. A 24-hour 4090 allocation is USD 17.76 in GPU charges; use a USD 25–35 planning envelope for this stage, not a guaranteed invoice. Storage, transfer and applicable taxes depend on provider/configuration.

[Runpod pricing](https://www.runpod.io/pricing). [Vast marketplace pricing](https://vast.ai/pricing). Vast individual host offers require a fresh quote. Choose a 5090 only if software compatibility and measured cost per completed workload justify it; its rate requires >1.34× 4090 speed to be cheaper on compute.

A larger GPU does not itself improve validity. No multi-GPU/FSDP complexity for this pilot. Free Colab may serve smoke checks; the scientific matrix uses a fixed BF16-capable GPU model and pinned stack. A hardware/precision change becomes a documented new stratum; never pool it silently.

## Benchmark

Benchmark RMS, taper-intermediate, zero-gate taper and optional auxiliary training, with real packed data. Include 50 warmup and at least 200 timed updates per materially different phase if the two-hour setup cap permits. Synchronize timed CUDA regions. Benchmark compiled/uncompiled once, choose globally before runs, and record compilation startup separately.

Time full-dev, quick-dev, diagnostic forward passes and one actual checkpoint write. Project the exact event schedules, not a generic percent overhead. If no allocation fits, stop feasibility and report measured cost; don't delete controls or switch model size after seeing results.

## Correctness checks

FP32 RMS endpoint parity and fold equivalence on deterministic small tensors/logits: rtol=1e-5, atol=1e-6. Investigate violations; record measured maximum errors. BF16 comparisons get separately documented tolerances after FP32 passes.

Confirm gradients through both active branches; gamma-tilde changes after activation and no internal division occurs at zero gate. Verify final RMSNorm retained, zero gain weight decay, correct EMA update frequency, aux omitted from evaluation and aux onset at 764.

Validate EMA implementation against offline reconstruction on the same recorded update means (FP32 rtol=1e-5). A later snapshot is a distribution diagnostic, not a 2% equality test.

Check microbatch accumulation equivalence on deterministic small batches, CE label shift, and no cross-split packing.

Compare ten uninterrupted/resumed updates in a deterministic test mode; compare weights, Adam moments, optimizer step, LR/gate, RNG and data cursor. Do not claim bitwise equivalence under kernels known to be nondeterministic.

## Logging and checkpoints

Record every run: run ID, protocol/config/code hashes, seed, condition, branch, parent checkpoint hash, hardware/driver/CUDA/library versions, precision, compilation, start/end time, completed updates, supervised tokens, actual GPU-hours, costs and outcome.

At every full-dev point save weight-only FP32 state and per-document/per-class sums/counts. At switch and every branch endpoint save complete resumable state; rotate full checkpoints every 15 minutes.

Complete state includes all weights, optimizer parameter groups/moments/per-parameter step counts, scaler if any, EMA buffers/counters, c and targets, LR/gate clock, all RNG state, packed-data cursor/order hashes, schema/version and code/config references.

Keep prefix checkpoints durably outside an ephemeral rental disk. Upload and verify checksum before terminating a rental. Store raw text only where authorized; Notion run records link manifests/results and document IDs, not raw corpus content.

## Failure policy

Never replace a poor finite-loss seed with another seed. Infrastructure interruption resumes the same state. Numerical/data/code defects require a documented cause and repair; mark affected results invalid and retain them. Shared scientific fixes require consistent reruns of affected conditions within the cap; otherwise report incomplete.

Set an automated wall-clock/budget watchdog that stops safely and saves state before exhausting the cap. It is part of implementation, not an automation already created.

## Manifest and locking

Before seed 101 record actual dataset revisions, tokenizer hashes, source code commit, container/package digest, fixed model/batch, allocation, GPU model, event lists and analysis hash. Save a timestamped immutable copy in the code repository or durable artifact store. Notion remains the navigable specification, not the only provenance record.

All run statuses begin Not started or Conditional. No empirical gate is marked passed merely because this project exists.