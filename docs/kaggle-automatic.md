# Automatic H1 checkpoint workflow

Use `artifacts/kaggle-auto/kaggle-automatic.ipynb` with `domain-shift-auto.zip`.
The notebook has three code cells: settings/input discovery, pinned runtime, and
automatic execution. Run All after filling the current session and weekly quota
values. No MODE or manual preflight/start/resume transitions are used.

## One-time setup

Select T4 x2, enable Internet, and create a notebook Secret named
`KAGGLE_API_TOKEN` (Add-ons -> Secrets), enabled for the notebook. The token comes
from https://www.kaggle.com/settings/api. A local `.env` is for local testing only;
it is never packaged or uploaded. Do not attach it to Kaggle as a dataset.

Attach the new automatic code bundle, the existing prepared online corpus, and
the recovered old notebook Output if it contains checkpoints. Do not attach the
old code bundle alongside the new one. All files under `src` in the automatic
bundle are byte-identical to the original paired bundle, preserving checkpoint
identity. The additional driver is under `scripts`, outside the frozen source.
Keep the backup dataset name constant on future runs.

The default private backup destination is
`danny00/domain-shift-h1-checkpoints-auto`. Only one active notebook may use it.
The driver never writes to unrelated datasets or makes a dataset public. Local
validation used a separate tiny private dataset,
`danny00/domain-shift-auto-transport-test`, without project data.

## What happens automatically

1. An existing remote run takes precedence. API/authentication failures stop the
   run; they do not mean “no checkpoint” and cannot trigger a fresh experiment.
2. Without remote state, full resume archives or extracted Output in Input or Working are
   inspected. The latest archive of the same experiment is chosen by its ledger.
   Multiple distinct experiments, corrupt archives, and failed scientific runs
   require investigation rather than silently choosing another seed.
3. A recovered run is verified, backed up remotely, downloaded and restored before
   continuation. Its existing weights and optimizer/RNG/data-position state stay
   intact. Its preflight is not repeated.
4. Only a genuinely new run performs preflight. The full preflight checkpoint is
   uploaded, independently downloaded, checksum-verified and replayed for five
   updates in fresh worker processes. Training then starts in the same session.
5. About every 30 work minutes, and earlier at prefix boundaries, both workers
   stop cleanly. Their joint checkpoint is uploaded to a new private dataset
   version, downloaded independently, verified and restored. Prefix backup
   receipts are registered from that verified remote copy. Training continues
   automatically while sufficient session/budget time remains.
6. A future Run All retrieves the latest remote state without manually attaching
   another checkpoint. A completed experiment is not trained again.

This changes the engineering proof from a manually restarted session to a real
cloud round trip plus a fresh-process replay within the session. The five-update
comparison, training code, scientific allocation and 50 elapsed-hour cap remain.
The adapter changes only the persisted-source admission rule for the exact
remote-downloaded archive during import; local ZIPs alone are still insufficient.

## Persistence and storage

Kaggle unpacks files uploaded with `.zip` names. The private dataset therefore
stores each ZIP payload with a `.bin` name; its bytes and internal receipt remain
unchanged. Large downloads may have an additional server ZIP wrapper, which the
driver extracts only for the expected filename. The driver then verifies SHA-256
and the original checkpoint generation receipts.

Uploads and remote reads are synchronous. Failed uploads/read-back verification
stop continuation and retain the local pending checkpoint. The last confirmed
remote checkpoint remains in an older version. The driver does not delete remote
versions: older evaluation-weight snapshots may be needed for later analysis.
The remote history records version numbers and filenames for those snapshots.
Any historical artifacts already missing before adoption cannot be recreated;
keep the recovered old notebook Output and previously saved archives.

During normal execution only one restored run, one downloaded checkpoint, and
one temporary upload are needed. Previous local generations are removed only
after a complete remote round trip and successful restore. Full archives are in
managed `/tmp` scratch; Output contains `automatic-status.json` and
`latest-reports.zip`, not a growing stack of multi-gigabyte ZIPs. Error leftovers
are deliberately retained for diagnosis. Remote storage is finite: if Kaggle's
account storage or API limits are reached, backup fails closed; older versions
are not silently deleted to make room.

## Interruption, concurrency and compute accounting

A platform kill can lose work since the last confirmed remote checkpoint (normally
up to the current work chunk, plus in-flight backup time). No notebook can promise
zero loss between backups. Rerunning automatically restores the last verified
checkpoint. A remote time reservation conservatively charges interrupted work.
While that reservation is still active a second invocation refuses to train and
shows its expiry, protecting against common overlapping-session mistakes. A local
OS lock and remote version precondition checks add protection, but Kaggle dataset
versions are not a distributed compare-and-swap lock. Do not run two sessions for
the same experiment simultaneously.

Kaggle still controls accelerator quotas and session lifetime. This notebook does
not bypass those limits or automatically launch new GPU sessions. Start the same
notebook next session and Run All; progress recovery is automatic.

The remote budget includes imported parent-ledger usage, current setup, preflight,
training, transfers and verification. In-flight operations reserve time first;
successful completion reconciles it to measured elapsed time. A hard kill retains
the reservation. `EXTRA_UNRECORDED_HOURS` is optional known usage outside saved
ledgers (for example earlier idle time); don't enter already-recorded training.
The two current remaining-time fields constrain this session only. The scientific
cap remains 50 cumulative notebook hours across weekly quota resets.

## Validation and references

Local integration tests exercise private-store publication, complete read-back,
corruption rejection, both prefix receipts, historical-weight offload, discovery,
preflight-to-training, automatic repeated chunks, completed-run discovery,
concurrency preconditions and cleanup boundaries. Tiny live tests validate the
official Kaggle API on the user's private account. Full-size transfers and actual
two-T4 execution of the combined workflow still require the notebook run.

Official references:
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md
- https://www.kaggle.com/docs/notebooks
- https://docs.astral.sh/uv/concepts/python-versions/
- https://pytorch.org/get-started/previous-versions/
