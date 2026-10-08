"""OpenTimestamps stamp / upgrade / info for the Study 2 registration manifest (Protocol Amendment 1).

The official `ots` client fails to import on Windows (python-bitcoinlib looks for an OpenSSL DLL), so this does the
same two steps with the client's own core library: SHA-256 the file, append a random 16-byte nonce, hash again and
submit only that 32-byte digest to the public calendars (stamp); later fetch the calendars' Bitcoin attestations
(upgrade). The resulting `.ots` proof is standard and verifies with `ots verify` on any platform where it runs.

  .venv\\Scripts\\python study2\\timestamp.py stamp study2\\registration-manifest.json
  .venv\\Scripts\\python study2\\timestamp.py upgrade study2\\registration-manifest.json.ots   # hours later
  .venv\\Scripts\\python study2\\timestamp.py info study2\\registration-manifest.json.ots
"""
import argparse
import os
from pathlib import Path
import sys

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

CALENDARS = ("https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org",
             "https://a.pool.eternitywall.com", "https://ots.btc.catallaxy.com")  # the ots client's defaults
MINIMUM = 2
UPGRADE_DOMAINS = (".calendar.opentimestamps.org", ".calendar.eternitywall.com", ".calendar.catallaxy.com")


def trusted(uri: str) -> bool:
    """The ots client's default whitelist (https://*.calendar.<operator>) for fetching upgrades."""
    from urllib.parse import urlsplit
    parts = urlsplit(uri)
    return parts.scheme == "https" and any((parts.hostname or "").endswith(d) for d in UPGRADE_DOMAINS)


def stamp(path: Path) -> Path:
    target = path.with_name(path.name + ".ots")
    if target.exists():
        raise SystemExit(f"{target} exists; never overwrite a timestamp proof")
    with path.open("rb") as stream:
        detached = DetachedTimestampFile.from_fd(OpSHA256(), stream)
    tip = detached.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
    accepted = []
    for url in CALENDARS:
        try:
            tip.merge(RemoteCalendar(url).submit(tip.msg, timeout=30))
            accepted.append(url)
        except Exception as exc:  # one unreachable calendar is fine; fewer than MINIMUM is not
            print(f"{url}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if len(accepted) < MINIMUM:
        raise SystemExit(f"Only {len(accepted)} calendar(s) accepted the digest; need {MINIMUM}")
    with target.open("xb") as stream:
        detached.serialize(StreamSerializationContext(stream))
    print(f"SHA-256 {detached.file_digest.hex()} submitted to {accepted} -> {target}")
    return target


def load(proof: Path) -> DetachedTimestampFile:
    with proof.open("rb") as stream:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(stream))


def upgrade(proof: Path) -> bool:
    detached = load(proof)
    changed = False
    for sub, attestation in list(detached.timestamp.all_attestations()):
        if not isinstance(attestation, PendingAttestation) or not trusted(attestation.uri):
            continue
        try:
            sub.merge(RemoteCalendar(attestation.uri).get_timestamp(sub.msg, timeout=30))
            changed = True
        except Exception as exc:  # not yet in a Bitcoin block: try again later
            print(f"{attestation.uri}: not upgraded yet ({type(exc).__name__})", file=sys.stderr)
    complete = any(isinstance(a, BitcoinBlockHeaderAttestation) for _, a in detached.timestamp.all_attestations())
    if changed:
        temporary = proof.with_name(proof.name + ".tmp")
        with temporary.open("wb") as stream:
            detached.serialize(StreamSerializationContext(stream))
        os.replace(temporary, proof)
    print(f"{proof}: {'complete (Bitcoin attestation present)' if complete else 'pending'}")
    return complete


def info(proof: Path) -> None:
    detached = load(proof)
    print(f"File SHA-256: {detached.file_digest.hex()}")
    for _, attestation in detached.timestamp.all_attestations():
        print(f"  {attestation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["stamp", "upgrade", "info"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    {"stamp": stamp, "upgrade": upgrade, "info": info}[args.command](args.path)
