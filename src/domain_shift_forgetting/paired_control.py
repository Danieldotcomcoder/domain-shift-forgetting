"""CPU coordinator utilities: isolated workers, joint stopping and durable imports."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time

from .pilot_control import read_json, write_json

CONDITIONS = ("RMS", "Taper-minus")


@contextmanager
def exclusive_run(root):
    """OS locks release on process death; a second live coordinator cannot enter."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "coordinator.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"1")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another coordinator is already using this run") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def worker_environment(gpu):
    return os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                         "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}


def kill_worker_tree(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def supervise(jobs, root, stop_file, *, deadline, stop_on_pause=False, grace_seconds=240):
    """Join every writer before return/export, even after failure or interruption."""
    root.mkdir(parents=True, exist_ok=True)
    processes, outputs, offsets, announced, lifetimes = {}, {}, {}, set(), {}
    stopped_at = None
    previous = {}
    active = root / "writers-active.json"
    write_json(active, {"coordinator_pid": os.getpid(), "workers": list(jobs)})

    def stop(*_):
        nonlocal stopped_at
        if stopped_at is None:
            stopped_at = time.monotonic()
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.touch()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        for name, job in jobs.items():
            log = root / f"{name}.log"
            outputs[name] = log.open("ab", buffering=0)
            offsets[name] = log.stat().st_size
            processes[name] = subprocess.Popen(job["command"], env=job["env"], stdout=outputs[name],
                stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0)
            lifetimes[name] = {"started": time.monotonic()}
        while True:
            for name, process in processes.items():
                with (root / f"{name}.log").open("rb") as stream:
                    stream.seek(offsets[name])
                    chunk = stream.read(8192)
                    offsets[name] += len(chunk)
                if chunk:
                    print(f"[{name}] {chunk.decode('utf-8', errors='replace')}", end="", flush=True)
                result = process.poll()
                if result is not None and name not in announced:
                    announced.add(name)
                    lifetimes[name]["elapsed_seconds"] = time.monotonic() - lifetimes[name]["started"]
                    if result != 0:
                        stop()
                    elif stop_on_pause:
                        status = read_json(jobs[name]["status"]) if jobs[name]["status"].exists() else {}
                        if status.get("status") != "completed":
                            stop()
            if all(p.poll() is not None for p in processes.values()):
                break
            if time.monotonic() >= deadline:
                stop()
            if stopped_at is not None and time.monotonic() - stopped_at >= grace_seconds:
                for p in processes.values():
                    if p.poll() is None:
                        kill_worker_tree(p)
            time.sleep(0.2)
        results = {name: p.wait() for name, p in processes.items()}
        if any(code != 0 for code in results.values()):
            raise RuntimeError(f"Worker failed; both writers joined. Inspect logs: {results}")
        return results
    finally:
        # Popen failure, unexpected controller exception, or a second interrupt.
        stop()
        for p in processes.values():
            if p.poll() is None:
                try:
                    p.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    kill_worker_tree(p)
                    p.wait()
        for handle in outputs.values():
            handle.close()
        for row in lifetimes.values():
            row.setdefault("elapsed_seconds", time.monotonic() - row["started"])
            row.pop("started", None)
        write_json(root / "worker-lifetimes.json", lifetimes)
        active.unlink()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def assert_quiescent(root):
    if any(root.rglob("writers-active.json")):
        raise ValueError("Full export refused: writers have not been confirmed stopped and joined")
    ledger = root / "notebook-ledger.json"
    if ledger.exists() and any(row["status"] == "running" for row in read_json(ledger)["sessions"]):
        raise ValueError("Full export refused: coordinator ledger is still running/unclean")


def barrier(root, label, condition, clock, stop_file):
    """All timed phases start together; a peer failure releases waiters."""
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    (folder / condition).touch()
    while not all((folder / name).exists() for name in CONDITIONS):
        if stop_file.exists() or clock.must_stop(30):
            raise TimeoutError("Peer/deadline interrupted the paired preflight")
        time.sleep(0.2)
    if stop_file.exists():
        raise TimeoutError("Paired preflight stopped")


def propagate_archive_indexes(root):
    """Global archive indexes are projected into each isolated worker's namespace."""
    durable = read_json(root / "durable-prefixes.json")
    history = read_json(root / "archived-weights.json") if (root / "archived-weights.json").exists() else {}
    for condition in CONDITIONS:
        child = root / "workers" / condition
        if not child.is_dir():
            raise ValueError("Paired archive is missing a worker")
        prefix = f"workers/{condition}/"
        normalized = {key.replace("\\", "/"): value for key, value in durable["prefixes"].items()}
        write_json(child / "durable-prefixes.json", durable | {"prefixes": {
            str(Path(key[len(prefix):])): value for key, value in normalized.items() if key.startswith(prefix)}})
        write_json(child / "archived-weights.json", {
            key[len(prefix):]: value for key, value in history.items() if key.startswith(prefix)})
