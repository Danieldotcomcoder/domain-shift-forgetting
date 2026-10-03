"""Automatic Kaggle orchestration. Kept OUTSIDE src to preserve frozen run identities.

Transport is private, versioned Kaggle Datasets. Successful uploads are downloaded
again and hashed before prefixes are acknowledged or temporary files are removed.
No model, optimizer, sampler, seed, event schedule or scientific rule is changed.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    from domain_shift_forgetting.pilot_control import write_json
    write_json(Path(path), data)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def member(source, name):
    if source.is_dir():
        return read(source / name)
    with zipfile.ZipFile(source) as z:
        return json.loads(z.read(name))


def discover(inputs, working=None):
    """Never silently start over when any recognizable prior run is attached."""
    roots = [inputs] + ([working] if working is not None and working.exists() else [])
    sources = list({p for base in roots for p in base.rglob("paired-resume-*.zip") if "reports" not in p.name})
    sources += [p.parent for base in roots for p in base.rglob("freeze.json")
                if (p.parent / "workers/RMS/freeze.json").exists()
                and (p.parent / "archive-receipt.json").exists()]
    rows = []
    for source in sources:
        freeze = member(source, "freeze.json")
        if freeze.get("layout") != "paired-v1":
            raise ValueError(f"Unexpected run format: {source}")
        receipt = member(source, "archive-receipt.json")
        if receipt.get("kind") != "resume":
            raise ValueError(f"Not a full checkpoint: {source}")
        ledger = member(source, "notebook-ledger.json")
        rank = max((s["started_utc"] for s in ledger["sessions"]), default="")
        identity = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
        rows.append((rank, identity, source))
    if not rows:
        return None
    if len({r[1] for r in rows}) != 1:
        raise ValueError("Multiple different experiments attached; keep only the intended experiment's outputs.")
    rows.sort(key=lambda r: (r[0], str(r[2])))
    return rows[-1][2]


def stage_candidate(source, destination):
    """Verify recovered Input OR Working output, without treating it as a cloud backup.

The staged run is never passed to training: save_and_restore must first complete
the network round trip. Clear any locally minted prefix receipts immediately.
"""
    from domain_shift_forgetting import pilot_control as pc
    original = pc.require_persisted_input
    expected = sha(source / 'archive-receipt.json' if source.is_dir() else source)
    def require_same_candidate(path):
        if Path(path).resolve() != source.resolve() or sha(source / 'archive-receipt.json' if source.is_dir() else source) != expected:
            raise ValueError('Recovered checkpoint changed during staging')
    pc.require_persisted_input = require_same_candidate
    try:
        (pc.import_directory if source.is_dir() else pc.import_run)(source, destination)
    finally:
        pc.require_persisted_input = original
    for receipt in destination.rglob('durable-prefixes.json'):
        write(receipt, {'prefixes': {}, 'status':'staged only; remote verification required'})


def ledger_hours(root):
    path = root / "notebook-ledger.json"
    if not path.exists():
        return 0.0, 0.0
    ledger = read(path)
    charged = sum(s["charged_seconds"] for s in ledger["sessions"]) / 3600
    return ledger["external_gpu_hours"] + charged, charged


def validate_remote(archive, proof):
    if (proof.get("schema") != "kaggle-readback-v1" or proof.get("verified") is not True
        or not re.fullmatch(r"[\w-]+/[\w-]+", proof.get("dataset", ""))
        or not isinstance(proof.get("version"), int) or proof["version"] < 1
        or proof.get("sha256") != sha(archive)):
        raise ValueError("Missing or mismatched remote read-back verification")


def import_verified(archive, destination, proof):
    """Use the original byte/receipt/checkpoint verifier with a scoped cloud-source adapter.

The source rule is replaced ONLY for the exact hash-verified downloaded archive,
then restored, even on failure. Ordinary local ZIPs remain inadmissible.
"""
    from domain_shift_forgetting import pilot_control as pc
    from domain_shift_forgetting.paired_control import propagate_archive_indexes
    validate_remote(archive, proof)
    original = pc.require_persisted_input

    def require_downloaded(path):
        if Path(path).resolve() != archive.resolve():
            raise ValueError("Unexpected restore source")
        validate_remote(archive, proof)

    pc.require_persisted_input = require_downloaded
    try:
        pc.import_run(archive, destination)
    finally:
        pc.require_persisted_input = original
    propagate_archive_indexes(destination)
    for condition in ("RMS", "Taper-minus"):
        write(destination / "workers" / condition / "automatic-remote-proof.json",
              proof | {"archive": str(archive.resolve())})


class KaggleStore:
    """All failures other than authenticated HTTP 404 are fatal, never 'new run'."""
    def __init__(self, dataset, cache, api=None, timeout=1200):
        if not re.fullmatch(r"[\w-]+/[\w-]+", dataset):
            raise ValueError("Dataset must be owner/slug")
        self.dataset, self.cache, self.timeout = dataset, Path(cache), timeout
        self.cache.mkdir(parents=True, exist_ok=True)
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        self.api = api

    def info(self):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = self.dataset.split("/")
        try:
            with self.api.build_kaggle_client() as client:
                result = client.datasets.dataset_api_client.get_dataset(request)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404 or getattr(exc, "status", None) == 404:
                return None
            if status == 403:
                # Kaggle hides nonexistent private slugs behind 403. Prove absence
                # via an authenticated, exhausted listing of OUR owned datasets;
                # never treat the 403 alone as permission to start over.
                owner, seen_owner = self.dataset.split('/')[0], False
                for page in range(1, 101):
                    rows = self.api.dataset_list(mine=True, page=page) or []
                    for row in rows:
                        ref = getattr(row, 'ref', '')
                        if ref == self.dataset:
                            raise RuntimeError("Checkpoint dataset exists but access was denied; stopping.") from None
                        seen_owner |= ref.startswith(owner + '/')
                    if not rows:
                        if seen_owner:
                            return None
                        break
            raise RuntimeError("Cannot read checkpoint dataset; check Secret/access/network. No new run was started.") from None
        if result.is_private is not True:
            raise ValueError("Checkpoint dataset must be private; refusing to upload.")
        return int(result.current_version_number)

    def download(self, version, name, directory):
        if Path(name).name != name:
            raise ValueError("Unsafe remote file name")
        directory.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(io.StringIO()):
            self.api.dataset_download_file(f"{self.dataset}/{version}", name,
                path=str(directory), force=True, quiet=True)
        path = directory / name
        # Kaggle wraps large single-file downloads in an extra ZIP. Extract only
        # the requested basename, never arbitrary paths from an archive.
        wrapper = directory / (name + '.zip')
        if not path.is_file() and wrapper.is_file():
            with zipfile.ZipFile(wrapper) as z:
                if z.namelist() != [name]:
                    raise ValueError('Unexpected remote download wrapper')
                with z.open(name) as incoming, path.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            wrapper.unlink()
        if not path.is_file():
            raise ValueError(f"Remote file not downloaded: {name}")
        return path

    def head(self):
        version = self.info()
        if version is None:
            return None
        with tempfile.TemporaryDirectory(dir=self.cache, prefix="head-") as tmp:
            head = read(self.download(version, "head.json", Path(tmp)))
        if head.get("schema") != "h1-auto-v1":
            raise ValueError("Existing dataset is not this application's checkpoint store; refusing overwrite.")
        head["version"] = version
        return head

    def publish(self, head, archive=None, expected=None):
        current = self.info()
        if current != expected:
            raise RuntimeError("Checkpoint store changed: another notebook may be running. Stopping.")
        version = (current or 0) + 1
        record = dict(head, schema="h1-auto-v1", version=version, transaction=uuid.uuid4().hex)
        with tempfile.TemporaryDirectory(dir=self.cache, prefix="upload-") as tmp:
            folder = Path(tmp)
            write(folder / "dataset-metadata.json", {"id": self.dataset,
                "title": "Private H1 automatic checkpoints", "licenses": [{"name": "other"}]})
            if archive is not None:
                # Kaggle auto-extracts uploaded .zip filenames. Keep archive bytes
                # opaque with .bin; ZipFile still verifies them after download.
                name = f"checkpoint-{record['transaction']}.bin"
                try:
                    os.link(archive, folder / name)
                except OSError:
                    shutil.copyfile(archive, folder / name)
                record["checkpoint"] = {"file": name, "sha256": sha(archive), "version": version}
                record["history"] = list(head.get("history", [])) + ([head["checkpoint"]] if head.get("checkpoint") else [])
            write(folder / "head.json", record)
            if current is None:
                response = self.api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
            else:
                response = self.api.dataset_create_version(str(folder), "Automatic H1 checkpoint",
                    quiet=True, convert_to_csv=False, delete_old_versions=False)
            if getattr(response, "error", None):
                raise RuntimeError("Kaggle rejected checkpoint publication; local files retained.")
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                try:
                    actual = self.head()
                    if actual and actual.get("transaction") == record["transaction"]:
                        if actual["version"] != version:
                            raise RuntimeError("Unexpected dataset version; concurrent writers are unsupported.")
                        return actual
                    if actual and actual["version"] > version:
                        raise RuntimeError("Another checkpoint writer changed the dataset.")
                except RuntimeError:
                    raise
                except Exception:
                    pass  # Dataset creation/indexing may still be processing.
                print("Waiting for Kaggle to publish checkpoint metadata...", flush=True)
                time.sleep(10)
        raise TimeoutError("Checkpoint upload was not confirmed. Local files retained; do not train further.")

    def fetch_checkpoint(self, head, directory):
        row = head.get("checkpoint")
        if not row:
            raise ValueError("No checkpoint recorded")
        archive = self.download(row["version"], row["file"], directory)
        if sha(archive) != row["sha256"]:
            raise ValueError("Remote checkpoint checksum mismatch; local state retained.")
        proof = {"schema": "kaggle-readback-v1", "dataset": self.dataset,
                 "version": row["version"], "sha256": row["sha256"], "verified": True}
        return archive, proof


def replay_remote(preflight_root, data_root, orders_root, output):
    """Original five-update replay, in a new process, after a real cloud round trip."""
    import numpy as np
    import torch
    from domain_shift_forgetting.pilot_preflight import compare
    from domain_shift_forgetting.pilot_runner import source_identity, runtime_identity, make_state, ordered_update
    from domain_shift_forgetting.pilot_arrays import PilotArrays, load_orders
    from domain_shift_forgetting.local_training import restore
    receipt = read(preflight_root / "automatic-remote-proof.json")
    validate_remote(Path(receipt["archive"]), receipt)
    proof, base = read(preflight_root / "resume-proof.json"), read(preflight_root / "validation.json")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if base["code"] != source_identity() or base["runtime"] != runtime_identity(device):
        raise ValueError("Remote replay runtime/code differs")
    arrays = PilotArrays(data_root)
    if arrays.identity != base["arrays_sha256"] or sha(preflight_root / "resume-expected.pt") != proof["expected_sha256"]:
        raise ValueError("Remote replay data/checkpoint mismatch")
    orders = load_orders(orders_root, arrays, 101)
    model, optimizer, scaler, sampler = make_state(proof["seed"], proof["identity"]["condition"], proof["policy"], device)
    progress = restore(preflight_root / "resume.pt", model, optimizer, scaler, sampler, proof["identity"])
    cursor = progress["cursor"]
    for _ in range(proof["steps"]):
        ordered_update(model, optimizer, scaler, arrays.streams["web_train"], orders["web"][cursor:cursor + 32].astype(np.int64), proof["policy"], device)
        cursor += 32
    expected = torch.load(preflight_root / "resume-expected.pt", map_location="cpu", weights_only=True)
    compare({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "sampler": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, expected)
    result = {k: base[k] for k in ("code", "runtime", "arrays_sha256", "policy_sha256")}
    result.update(status="passed", external_checkpoint_replay=True,
                  source_proof_sha256=sha(preflight_root / "resume-proof.json"), remote=receipt,
                  mechanism="cloud read-back and fresh-process replay within session")
    write(output, result)


@contextmanager
def remote_replay_command():
    """Adapt ONLY the preflight replay subprocess; unchanged training workers stay original."""
    from domain_shift_forgetting import paired
    original = paired.command
    def command(action, **kwargs):
        if action != "_verify":
            return original(action, **kwargs)
        return [sys.executable, str(Path(__file__).resolve()), "--verify",
                *[str(kwargs[k]) for k in ("root", "data", "orders", "output")]]
    paired.command = command
    try:
        yield
    finally:
        paired.command = original


def safe_remove(path, managed):
    path, managed = Path(path).resolve(), Path(managed).resolve()
    if path == managed or not path.is_relative_to(managed):
        raise ValueError("Cleanup outside managed scratch refused")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def archive_clean(root, output):
    from domain_shift_forgetting.paired_control import assert_quiescent, exclusive_run
    from domain_shift_forgetting.pilot_control import export_run
    with exclusive_run(root):
        assert_quiescent(root)
        if (root / "failure.json").exists() or any(
            "test-tmp" not in p.relative_to(root).parts
            for p in root.rglob("IMPORT-FAILED.json")
        ):
            raise ValueError("Failed run requires investigation; refusing to publish it as resumable.")
        export_run(root, output)


def run_auto(args):
    from domain_shift_forgetting import paired
    from domain_shift_forgetting.paired_control import exclusive_run, propagate_archive_indexes
    from domain_shift_forgetting.pilot_control import import_run, import_directory, export_run, digest_json
    from domain_shift_forgetting.pilot_runner import source_identity
    # One local coordinator per dataset; remote precondition checks detect common collisions.
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    store = KaggleStore(args.dataset, work / "transport")
    policy = paired.policy_for_pair(args.config)
    if not (0 < args.minutes <= 720 and 0 < args.weekly_hours <= 100 and 0 <= args.prior_hours < 50):
        raise ValueError("Enter current session/quota hours and nonnegative prior overhead")
    began = time.monotonic()
    deadline = began + min(args.minutes, args.weekly_hours * 60) * 60
    def remaining():
        return (deadline - time.monotonic()) / 60

    with exclusive_run(work / "automatic-lock"):
        head = store.head()
        source = None if head else discover(args.inputs, Path('/kaggle/working'))
        root = None
        fingerprint = {"code": source_identity(), "policy": digest_json(policy),
                       "data": sha(args.data / "manifest.json"), "orders": sha(args.orders / "manifest.json")}
        if head:
            if head.get("fingerprint") != fingerprint:
                raise ValueError("Remote experiment uses different code/data/policy. Keep its original bundle; no restart performed.")
            if head.get("phase") == "failed":
                raise ValueError("Remote run records a failure; investigate it before resuming.")
            if head.get("active_until", 0) > time.time():
                expires = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(head['active_until']))
                raise ValueError(f"Previous operation reservation active until {expires}. Run again afterward; do not run two notebooks.")
            budget_base = head["budget_hours"] + args.prior_hours
            if head["phase"] == "adopting" and not head.get("checkpoint"):
                source = discover(args.inputs, Path('/kaggle/working'))
                if source is None:
                    raise ValueError("Recovered run backup was interrupted. Keep the recovered outputs attached; no new run started.")
                root = work / ("adopt-" + uuid.uuid4().hex)
                stage_candidate(source, root)
                if digest_json(read(root / "freeze.json")) != head["experiment"]:
                    raise ValueError("Recovered outputs changed experiment during backup")
        else:
            budget_base = args.prior_hours
            head = {"schema": "h1-auto-v1", "experiment": uuid.uuid4().hex,
                    "fingerprint": fingerprint, "phase": "initial", "budget_hours": budget_base,
                    "checkpoint": None, "history": [], "active_until": 0}
            if source:
                print(f"Recovering existing experiment from {source}", flush=True)
                root = work / ("adopt-" + uuid.uuid4().hex)
                stage_candidate(source, root)
                frozen = read(root / "freeze.json")
                if frozen["code"] != fingerprint["code"] or digest_json(frozen["policy"]) != fingerprint["policy"]:
                    raise ValueError("Recovered run needs its original code/policy. No new experiment started.")
                budget_base += ledger_hours(root)[0]
                head["experiment"] = digest_json(frozen)
                head["phase"] = "adopting"
                head["budget_hours"] = budget_base
                head["adopted_source"] = str(source)
            # Tiny private upload/read-back tests access before spending training quota.
            head = store.publish(head, expected=None)

        def hours():
            return budget_base + (time.monotonic() - began) / 3600

        def control(**changes):
            nonlocal head
            head = store.publish(head | changes, expected=head["version"])

        def reserve(minutes):
            if hours() + minutes / 60 > 50:
                raise ValueError("Insufficient remaining experiment budget for a safely backed-up chunk.")
            control(budget_hours=hours() + minutes / 60, active_until=time.time() + minutes * 60)

        def save_and_restore(current, phase, original_source=None):
            """Delete local generations ONLY after upload + separate download + verified import."""
            nonlocal head
            out = work / ("pending-" + uuid.uuid4().hex + ".zip")
            if original_source is None:
                archive_clean(current, out)
            else:
                # Imported history was offloaded by the legacy importer. Back up the
                # ORIGINAL archive on adoption so those weights are not discarded.
                from domain_shift_forgetting.paired_control import assert_quiescent
                assert_quiescent(current)
                if (current / "failure.json").exists():
                    raise ValueError("Recovered archive records a failed run; review before adoption.")
                if original_source.is_dir():
                    receipt = read(original_source / "archive-receipt.json")
                    with zipfile.ZipFile(out, 'x', zipfile.ZIP_STORED) as z:
                        for name in receipt['files']:
                            file = original_source / name
                            if not file.resolve().is_relative_to(original_source.resolve()) or sha(file) != receipt['files'][name]:
                                raise ValueError("Changed/unsafe attached checkpoint file")
                            z.write(file, name)
                        z.write(original_source / 'archive-receipt.json', 'archive-receipt.json')
                else:
                    try:
                        os.link(original_source, out)
                    except OSError:
                        shutil.copyfile(original_source, out)
            staged_head = head | {"phase": phase, "budget_hours": max(head["budget_hours"], hours() + 20 / 60)}
            head = store.publish(staged_head, archive=out, expected=head["version"])
            downloaded = work / ("download-" + uuid.uuid4().hex)
            cloud_zip, proof = store.fetch_checkpoint(head, downloaded)
            restored = work / ("restored-" + uuid.uuid4().hex)
            import_verified(cloud_zip, restored, proof)
            write(restored / "automatic-history.json", {"dataset": args.dataset, "archives": head["history"],
                  "latest": head["checkpoint"], "driver_sha256": sha(Path(__file__))})
            control(budget_hours=hours(), active_until=0)
            write(args.output / "automatic-status.json", head | {"last_verified_backup": proof})
            print(f"BACKUP VERIFIED: https://www.kaggle.com/datasets/{args.dataset}/versions/{head['checkpoint']['version']}", flush=True)
            safe_remove(out, work)
            safe_remove(current, work)
            # Keep the current read-back ZIP for fresh-process preflight replay. Remove older downloads.
            for old in work.glob("download-*"):
                if old != downloaded:
                    safe_remove(old, work)
            return restored

        if root is not None:
            reserve(25)
            root = save_and_restore(root, "training", original_source=source)
        elif head.get("checkpoint"):
            downloaded = work / ("download-" + uuid.uuid4().hex)
            archive, proof = store.fetch_checkpoint(head, downloaded)
            root = work / ("restored-" + uuid.uuid4().hex)
            import_verified(archive, root, proof)
        if head["phase"] == "completed":
            print("This experiment already completed. No additional training started.", flush=True)
            return

        common = dict(data=args.data, orders=args.orders, sources=args.data.parent / "upstream", config=args.config,
                      tests=args.tests, recover_unclean=0)
        if head["phase"] == "initial":
            if remaining() < 50:
                raise ValueError("Need at least 50 minutes for preflight and verified backup.")
            reserve(min(remaining(), 130))
            root = work / ("preflight-" + uuid.uuid4().hex)
            paired.preflight(SimpleNamespace(**common, root=root, remaining_minutes=remaining() - 20,
                                            external_notebook_hours=hours()))
            if read(root / "paired-preflight.json")["status"] != "passed":
                control(phase="failed", budget_hours=hours(), active_until=0, error="Preflight infeasible")
                raise ValueError("Preflight did not admit this allocation; training was not started.")
            root = save_and_restore(root, "preflight")
        if head["phase"] == "preflight":
            if remaining() < 30:
                print("Preflight safely backed up. Next Run All will continue automatically.", flush=True)
                return
            reserve(30)
            primary = work / ("training-" + uuid.uuid4().hex)
            with remote_replay_command():
                paired.start(SimpleNamespace(**common, root=primary, preflight=root,
                    remaining_minutes=remaining() - 20, external_notebook_hours=hours()))
            write(primary / "automatic-protocol.json", {"driver_sha256": sha(Path(__file__)),
                "amendment": "User requested automatic cloud round-trip and fresh-process proof in same session; numerical protocol unchanged."})
            safe_remove(root, work)
            root = save_and_restore(primary, "training")

        while remaining() > 30 and hours() + .5 < 50:
            work_minutes = min(args.chunk_minutes, remaining() - 30, (50 - hours()) * 60 - 30)
            if work_minutes < 1:
                break
            reserve(work_minutes + 30)
            # Include all setup/upload/idle and conservatively reserved lost work as external hours.
            _, training_hours = ledger_hours(root)
            external = max(0, hours() - training_hours)
            print(f"Continuing saved training; notebook budget {hours():.2f}/50 hours", flush=True)
            try:
                result = paired.run(SimpleNamespace(**common, root=root, remaining_minutes=work_minutes + 10,
                                                    external_notebook_hours=external))
            except BaseException:
                # Do not turn a correctness failure into a fresh seed or silently retry it.
                control(phase="failed", budget_hours=hours(), active_until=0,
                        error="Training failed; local logs retained. Review before continuing.")
                raise
            phase = "completed" if read(root / "status.json")["status"] == "completed" else "training"
            root = save_and_restore(root, phase)
            print(json.dumps({"decision": result.get("decision"), "budget_hours": hours()}), flush=True)
            report_zip = args.output / "latest-reports.zip"
            if report_zip.exists():
                report_zip.unlink()
            export_run(root, report_zip, reports_only=True)
            if phase == "completed":
                break
        control(budget_hours=hours(), active_until=0)
        write(args.output / "automatic-status.json", head)
        print("Saved automatically. Run this SAME notebook next session to continue; no ZIP handoff required.", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        replay_remote(*map(Path, sys.argv[2:]))
        return
    p = argparse.ArgumentParser()
    for key in ("inputs", "data", "config", "tests", "work", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--minutes", required=True, type=float)
    p.add_argument("--weekly-hours", required=True, type=float)
    p.add_argument("--prior-hours", type=float, default=0)
    p.add_argument("--chunk-minutes", type=float, default=30)
    args = p.parse_args()
    if not 1 <= args.chunk_minutes <= 60:
        p.error("chunk-minutes must be between 1 and 60")
    args.orders = args.data.parent / "orders"
    args.output.mkdir(parents=True, exist_ok=True)
    run_auto(args)


if __name__ == "__main__":
    main()
