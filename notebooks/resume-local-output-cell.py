# Replacement for the start/import cell only. Keep the training/export cell.
from pathlib import Path
import hashlib, json, shutil, tempfile, zipfile

source = Path('/kaggle/working/paired-resume-1790934440.zip')
assert source.is_file(), f'Archive not found: {source}'
restored = Path(tempfile.mkdtemp(prefix='paired-local-resume-')) / 'run'

# Verify the full archive and copy it into a fresh directory. Keep the original.
# Preserve existing durable-prefix receipts; a local ZIP is not an external backup.
with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    receipt = json.loads(archive.read('archive-receipt.json'))
    assert receipt.get('kind') == 'resume', 'Reports-only ZIPs cannot resume.'
    assert len(names) == len(set(names)), 'Duplicate archive members.'
    assert set(names) == set(receipt['files']) | {'archive-receipt.json'}, 'Invalid archive receipt.'
    for name in names:
        target = (restored / name).resolve()
        assert '\\' not in name and target.is_relative_to(restored.resolve()), 'Unsafe archive path.'
    freeze = json.loads(archive.read('freeze.json'))
    assert freeze.get('layout') == 'paired-v1' and freeze.get('admitted') is True, 'Not an admitted paired run.'
    for name, expected in receipt['files'].items():
        target = restored / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with archive.open(name) as incoming, target.open('wb') as outgoing:
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                outgoing.write(chunk)
        assert digest.hexdigest() == expected, f'Archive checksum mismatch: {name}'

# Use the existing checkpoint validator without modifying any frozen source code.
subprocess.run([sys.executable, '-c',
    'import sys; from pathlib import Path; '
    'from domain_shift_forgetting.pilot_control import checkpoint_members; '
    'from domain_shift_forgetting.paired_control import assert_quiescent; '
    'p=Path(sys.argv[1]); checkpoint_members(p); assert_quiescent(p)',
    str(restored)], env=env, check=True)
for condition in ('RMS', 'Taper-minus'):
    worker = restored / 'workers' / condition
    assert (worker / 'freeze.json').is_file(), f'Missing worker: {condition}'
    status = json.loads((worker / 'status.json').read_text())
    print(condition, status)
    if status.get('status') == 'awaiting_durable_prefix':
        print('This worker still needs an externally saved prefix before branching. Local restoration does not satisfy that check.')

root = restored
MODE = 'resume'
RUN_PRIMARY = True
print('Verified local archive:', source)
print('Restored run:', root)
print('Run the existing training/export cell next. No preprocessing or preflight is repeated.')
