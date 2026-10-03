"""Frozen admission, conservative session accounting and verified checkpoint export."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import zipfile

from .analysis.decision import DecisionThresholds
from .local_corpus import sha256

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")


def require_persisted_input(source: Path) -> None:
    if KAGGLE_WORKING.exists() and not source.resolve().is_relative_to(KAGGLE_INPUT.resolve()):
        raise ValueError("On Kaggle, restore the persisted archive from /kaggle/input, not the current ephemeral disk")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_policy(path: Path) -> dict:
    config = read_json(path)
    if config["seeds"] != [101, 102, 103] or config["conditions"] != ["RMS", "Taper-minus"]:
        raise ValueError("Primary allocation must retain all three paired seeds and both conditions")
    if type(config["microbatch"]) is not int or type(config["accumulation"]) is not int or min(config["microbatch"], config["accumulation"]) < 1 or config["microbatch"] * config["accumulation"] != 32:
        raise ValueError("Effective batch must be 32 sequences / 16,384 supervised tokens")
    if config["precision"] not in ("fp16", "bf16") or config["compile"] is not False:
        raise ValueError("Use the validated eager FP16/BF16 execution path")
    if config["precision"] == "fp16" and (not config["amendment"] or config["protocol"] == "v3"):
        raise ValueError("FP16 requires a named, explicit amendment")
    rules = DecisionThresholds(**config["decision_thresholds"])
    if asdict(rules) != asdict(DecisionThresholds()):
        raise ValueError("This original-protocol runner does not permit outcome threshold changes")
    for key in ("optimizer_hours_cap", "active_gpu_hours_cap", "checkpoint_interval_seconds",
                "session_chunk_minutes", "session_save_reserve_minutes", "numerical_max_ce_gap_nats"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid resource control: {key}")
    amendment = config.get("budget_amendment", {})
    amended = (config["protocol"] == "v3-kaggle-fp16-2"
               and amendment.get("id") == "kaggle-free-50h-20260930"
               and amendment.get("authorized_by") == "user"
               and amendment.get("active_gpu_hours_cap") == 50)
    paired = (config["protocol"] == "v3-kaggle-fp16-paired-1"
              and amendment.get("id") == "kaggle-paired-50-notebook-hours-20261001"
              and amendment.get("authorized_by") == "user"
              and config.get("execution", {}).get("budget_unit") == "elapsed_notebook_hours"
              and config["execution"].get("notebook_hours_cap") == 50
              and config["execution"].get("workers") == {"RMS": "0", "Taper-minus": "1"})
    if config.get("execution", {}).get("mode") == "paired" and not paired:
        raise ValueError("Invalid paired execution/budget amendment")
    optimizer_limit, total_limit = (50, 50) if amended or paired else (20, 24)
    if config["optimizer_hours_cap"] > optimizer_limit or config["active_gpu_hours_cap"] > total_limit:
        raise ValueError("Compute cap exceeds the documented budget amendment")
    if config["optimizer_hours_cap"] > config["active_gpu_hours_cap"]:
        raise ValueError("Optimizer time must fit inside the cumulative total budget")
    if config["checkpoint_interval_seconds"] > 900:
        raise ValueError("Full checkpoints must be no farther apart than 15 minutes")
    return config


@dataclass
class SessionClock:
    """Deadline is supplied from CURRENT remaining UI time, not a fresh 12 hours."""
    deadline: float
    reserve_seconds: float

    @classmethod
    def start(cls, remaining_minutes: float, chunk_minutes: float, reserve_minutes: float,
              *, now: float | None = None):
        if any(not math.isfinite(x) or x <= 0 for x in (remaining_minutes, chunk_minutes, reserve_minutes)):
            raise ValueError("Session minutes must be finite and positive")
        if remaining_minutes <= reserve_minutes:
            raise ValueError("Insufficient session time to checkpoint/export safely")
        return cls((time.monotonic() if now is None else now) + min(remaining_minutes, chunk_minutes + reserve_minutes) * 60,
                   reserve_minutes * 60)

    def must_stop(self, next_operation_seconds: float = 0, *, now: float | None = None) -> bool:
        if not math.isfinite(next_operation_seconds) or next_operation_seconds < 0:
            raise ValueError("Invalid operation estimate")
        return (time.monotonic() if now is None else now) + self.reserve_seconds + next_operation_seconds >= self.deadline


class SessionLedger:
    """Reserve a whole chunk before work. A killed process keeps that charge.

    Reconcile successful sessions down to measured time; never reset a failed run
    to zero. Platform/quota setup time outside this runner is supplied explicitly.
    Only one active GPU/process is supported by this protocol implementation.
    """
    def __init__(self, path: Path, *, reserved_seconds: float, external_gpu_hours: float,
                 cap_hours: float = 24, recover_unclean: bool = False):
        if not all(math.isfinite(v) and v >= 0 for v in (reserved_seconds, external_gpu_hours)) or reserved_seconds <= 0:
            raise ValueError("Invalid ledger allocation")
        self.path = path
        self.data = read_json(path) if path.exists() else {"schema": 1, "sessions": [], "external_gpu_hours": 0.0}
        if external_gpu_hours < self.data["external_gpu_hours"]:
            raise ValueError("Cumulative external usage cannot decrease")
        self.data["external_gpu_hours"] = external_gpu_hours
        for row in self.data["sessions"]:
            if row["status"] == "running":
                if not recover_unclean:
                    raise ValueError("Unclean/active prior session; confirm it has ended and use --recover-unclean")
                row["status"] = "unclean_reserved_time_charged"
                row["optimizer_seconds"] = max(row.get("optimizer_seconds", 0), row["charged_seconds"])
        billed = sum(row["charged_seconds"] for row in self.data["sessions"]) + external_gpu_hours * 3600
        available = cap_hours * 3600 - billed
        if available <= 0:
            raise ValueError("GPU budget exhausted; result is incomplete, not a scientific null")
        self.reserved = min(reserved_seconds, available)
        self.started = time.monotonic()
        self.row = {"id": uuid.uuid4().hex, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "charged_seconds": self.reserved, "optimizer_seconds": 0.0}
        self.data["sessions"].append(self.row)
        write_json(path, self.data)

    @property
    def previous_optimizer_seconds(self):
        return sum(row.get("optimizer_seconds", 0) for row in self.data["sessions"][:-1])

    def finish(self, optimizer_seconds: float, status: str):
        self.row.update(status=status, charged_seconds=time.monotonic() - self.started,
                        optimizer_seconds=optimizer_seconds)
        write_json(self.path, self.data)

    def record_optimizer(self, seconds: float):
        self.row["optimizer_seconds"] = seconds
        write_json(self.path, self.data)


def checkpoint_members(root: Path) -> list[Path]:
    """Collect authoritative generations; aliases duplicate hundreds of MB."""
    files = set()
    for pointer in root.rglob("*.pt.json"):
        if "test-tmp" in pointer.relative_to(root).parts:
            continue
        record = read_json(pointer)
        files.add(pointer)
        for entry in (record, record.get("previous")):
            if entry:
                name = entry["file"]
                if Path(name).name != name:
                    raise ValueError("Unsafe generation name")
                generation = pointer.parent / name
                if sha256(generation) != entry["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {generation}")
                files.add(generation)
    return sorted(files)


def export_run(root: Path, destination: Path, *, reports_only: bool = False) -> dict:
    """Export atomically with per-file hashes; original source remains untouched."""
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("Write archives outside the run directory")
    if destination.exists():
        raise FileExistsError("Choose a new archive name; retain previous durable evidence")
    files = {p for p in root.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".txt", ".md", ".xml", ".log")
             and not p.name.endswith(".pt.json") and "test-tmp" not in p.relative_to(root).parts
             and not (reports_only and (p.name == "training.jsonl" or "events" in p.relative_to(root).parts))}
    if not reports_only:
        files.update(checkpoint_members(root))
        files.update(p for p in root.rglob("weights-*.pt") if "test-tmp" not in p.relative_to(root).parts)
        files.update(p for p in root.rglob("resume-expected.pt") if "test-tmp" not in p.relative_to(root).parts)
    receipt = {"schema": 1, "kind": "reports" if reports_only else "resume",
               "files": {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("archive-receipt.json", json.dumps(receipt, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
    temporary.replace(destination)
    return {"archive": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}


def import_run(archive_path: Path, destination: Path) -> None:
    require_persisted_input(archive_path)
    if destination.exists():
        raise FileExistsError("Restore to a new run directory")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()) or "\\" in name:
                    raise ValueError("Unsafe archive member")
            receipt = json.loads(archive.read("archive-receipt.json"))
            if receipt["kind"] != "resume":
                raise ValueError("A reports-only archive cannot resume training")
            if set(archive.namelist()) != set(receipt["files"]) | {"archive-receipt.json"}:
                raise ValueError("Archive member list differs from receipt")
            archived_weights = {}
            for name, digest in receipt["files"].items():
                # Verify every byte, but keep historical weight-only snapshots
                # in the durable archive to avoid exhausting Kaggle's local disk.
                h = hashlib.sha256()
                skip = Path(name).name.startswith("weights-") and name.endswith(".pt")
                target = destination / name
                if not skip:
                    target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source:
                    sink = None if skip else target.open("wb")
                    try:
                        while chunk := source.read(1024 * 1024):
                            h.update(chunk)
                            if sink:
                                sink.write(chunk)
                    finally:
                        if sink:
                            sink.close()
                if h.hexdigest() != digest:
                    raise ValueError(f"Imported file hash mismatch: {name}")
                if skip:
                    archived_weights[name] = {"sha256": digest, "archive": archive_path.name}
        history_path = destination / "archived-weights.json"
        history = read_json(history_path) if history_path.exists() else {}
        history.update(archived_weights)
        write_json(history_path, history)
        checkpoint_members(destination)
        # The verified source archive is now the external copy of these prefixes.
        prefixes = {str(p.parent.relative_to(destination)): read_json(p)["sha256"]
                    for p in destination.rglob("switch.pt.json")}
        write_json(destination / "durable-prefixes.json", {"prefixes": prefixes,
            "archive": str(archive_path.resolve()), "archive_sha256": sha256(archive_path)})
    except Exception:
        # Keep evidence of a failed import; never make it look ready to resume.
        write_json(destination / "IMPORT-FAILED.json", {"status": "failed"})
        raise


def import_directory(source: Path, destination: Path) -> None:
    """Kaggle may unpack uploaded ZIPs. The receipt still authenticates each byte."""
    source = source.resolve()
    require_persisted_input(source)
    if destination.exists():
        raise FileExistsError("Restore to a new directory")
    receipt = read_json(source / "archive-receipt.json")
    if receipt.get("kind") != "resume":
        raise ValueError("Reports-only data cannot resume training")
    destination.mkdir(parents=True)
    archived_weights = {}
    try:
        import shutil
        for name, digest in receipt["files"].items():
            file = (source / name).resolve()
            if not file.is_relative_to(source) or "\\" in name:
                raise ValueError("Unsafe archive member")
            if sha256(file) != digest:
                raise ValueError(f"Persisted file hash mismatch: {name}")
            if Path(name).name.startswith("weights-") and name.endswith(".pt"):
                archived_weights[name] = {"sha256": digest, "archive": str(source),
                                           "receipt_sha256": sha256(source / "archive-receipt.json")}
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
        history_path = destination / "archived-weights.json"
        history = read_json(history_path) if history_path.exists() else {}
        history.update(archived_weights)
        write_json(history_path, history)
        checkpoint_members(destination)
        prefixes = {str(p.parent.relative_to(destination)): read_json(p)["sha256"] for p in destination.rglob("switch.pt.json")}
        write_json(destination / "durable-prefixes.json", {"prefixes": prefixes, "archive": str(source),
            "archive_is_directory": True, "archive_sha256": sha256(source / "archive-receipt.json")})
    except Exception:
        write_json(destination / "IMPORT-FAILED.json", {"status": "failed"})
        raise
