# Study 2 execution log

All times are UTC. GPU quota is Kaggle T4 × 2, billed at 1× elapsed. At 21:54 on 8 October the author reported
18 GPU hours available, with the weekly reset to 30 h about 26 h later (around 00:00 UTC on 10 October).

| Time (UTC) | Step | Notes |
|---|---|---|
| 2026-10-08 ~21:50 | Author decisions | Run now with a hash-timestamp first (Amendment 1); design approved as written (commit `0b3a2ce`) |
| 2026-10-08 21:54 | Docker pin | `pin-image`: `gcr.io/kaggle-private-byod/python@sha256:2757e0c7…` (pilot run notebook) |
| 2026-10-08 21:54 | Freeze | `study2/registration-manifest.json` (commit `fa09196`, files at `321e783`), SHA-256 `3eeda992819ae9428a00c6a98688cf9b172879a136c771d7b9794ed92110e45a` |
| 2026-10-08 21:55 | OpenTimestamps | Digest accepted by 4 calendars; proof `study2/registration-manifest.json.ots` (commit `bc515cd`), pending Bitcoin attestation |
| 2026-10-08 21:56 | Probe pushed | `danny00/s2-domain-probe` version 1 (GPU T4 × 2, internet on). First Study 2 job |
| 2026-10-08 ~22:01 | Probe complete | About 4 min; torch 2.11.0+cu128 (pinned image). Reproduction of the pilot's switch energies: max relative difference **0.0** (gate passed). Selection **mc4-zh**, S = 0.0934 (RMS 0.0757, Taper-minus 0.1111). Ranking zh 0.0934 > ru 0.0893 > de 0.0514 > OpenWebMath 0.0119. Python reference 0.1099. **No candidate exceeds Python** (Protocol 4.6 applies: proceed, with the premise qualification on H2). Records in `reports/s2-kaggle/probe/` |
| 2026-10-08 22:01 | Preparation pushed | `danny00/s2-prepare-domain` version 1 (CPU, internet on); attaches the probe output and the pilot dataset |
| 2026-10-08 22:10 | Preparation complete | 7.6 min, **accepted** (details below). Records in `reports/s2-kaggle/prepare/` |

Preparation details:
- **Sources.** Train shards `c4-zh.tfrecord-00314` and `-00305` of 01024; validation shard `c4-zh-validation.tfrecord-00001` of 00002. All SHA-256-verified against the hub.
- **Raw tokens before deduplication.** Train 126.5M, dev 9.37M, test 9.66M; one filter round.
- **Deduplication.** 698 near-duplicates removed; 0 exact duplicates; 0 duplicates against the frozen pilot corpus. Audit: 0 of 2,443 cross-split pairs and 0 of 2,000 frozen pairs at or above 0.85.
- **Materialized.** X train 110,000,129 tokens (56,276 documents); X dev 2,097,153 (1,139 documents); reserved test 8,388,609 (4,201 documents, sealed).
- **Orders.** The pilot's seed 101–103 orders were reproduced bit for bit (NumPy 2.1.3 on Kaggle).
- **Manifest hashes.** Online manifest `5994e6fb…`; orders manifest `ba2596ce…`.
