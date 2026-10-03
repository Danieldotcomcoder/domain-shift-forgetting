"""Build the single automatic notebook and an explicitly allowlisted code bundle."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/kaggle-auto'
out.mkdir(parents=True, exist_ok=True)

def cell(kind, text):
    result = {'cell_type':kind, 'metadata':{}, 'source':text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

intro = '''# H1: automatic backup and resume

Use **Run All**. If a checkpoint exists in the private backup dataset, this resumes it. Otherwise it adopts the latest recovered full checkpoint attached to Input. Only if neither exists does it run preflight and then start training automatically. Seeds, data, effective batch, precision and scientific thresholds are unchanged.

One-time setup:
1. Select **T4 x2**, enable Internet, and add a Kaggle Secret named **KAGGLE_API_TOKEN** with notebook access enabled. Never put the token in a code cell or dataset.
2. Attach **domain-shift-auto.zip**, the existing **stage1-online-inputs** dataset, and your recovered notebook Output (if available). Do not also attach the old code bundle.
3. Fill the two time values in Cell 1. They are current remaining amounts, not nominal limits. Run only one notebook for this checkpoint dataset at a time.

No MODE, manual preflight/start switch, ZIP downloads, or checkpoint reattachments are needed after the first automatic backup. Ordinary notebook sessions still have to be started by you. If Kaggle kills a session, the next Run All restores the last verified remote checkpoint. Work after that checkpoint may need to be repeated; reserved time remains charged conservatively.

The original training source is unchanged, including checkpoint/replay validation. This external driver replaces the manual backup handoff with a private cloud upload, independent download and hash verification. First-run preflight recovery uses a fresh process in the same session, rather than a manually started session.
'''
setup = '''from pathlib import Path
import os, sys, json, subprocess, tempfile, time, zipfile, signal

# SAME dataset on every future run. Changing it creates a different backup store.
CHECKPOINT_DATASET = 'danny00/domain-shift-h1-checkpoints-auto'
SESSION_REMAINING_MINUTES = None  # enter the current live-session remaining minutes
WEEKLY_GPU_HOURS_REMAINING = None  # enter the current remaining GPU quota
EXTRA_UNRECORDED_HOURS = 0.0  # optional known prior idle/setup time not in saved ledgers
for value in (SESSION_REMAINING_MINUTES, WEEKLY_GPU_HOURS_REMAINING):
    assert isinstance(value, (int, float)) and value > 0, 'Fill both remaining-time values first.'
setup_started = time.monotonic()

from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
assert token.strip(), 'Enable access to the KAGGLE_API_TOKEN notebook Secret.'
inputs = Path('/kaggle/input')
drivers = list(inputs.rglob('scripts/kaggle_auto.py'))
if drivers:
    assert len(drivers) == 1, 'Attach exactly one automatic code bundle.'
    code = drivers[0].parent.parent
else:
    bundles = list(inputs.rglob('domain-shift-auto.zip'))
    assert len(bundles) == 1, 'Attach domain-shift-auto.zip.'
    code = Path(tempfile.mkdtemp(prefix='h1-auto-code-'))
    with zipfile.ZipFile(bundles[0]) as z:
        for name in z.namelist():
            assert '\\\\' not in name and (code / name).resolve().is_relative_to(code.resolve())
        z.extractall(code)
data_roots = [p.parent for p in inputs.rglob('manifest.json') if p.parent.name == 'online'
              and json.loads(p.read_text()).get('schema') == 'pilot-arrays-v1']
if not data_roots:
    bundles = list(inputs.rglob('stage1-online-inputs.zip'))
    assert len(bundles) == 1, 'Attach the existing prepared online inputs.'
    target = Path(tempfile.mkdtemp(prefix='h1-auto-inputs-'))
    with zipfile.ZipFile(bundles[0]) as z:
        for name in z.namelist():
            assert '\\\\' not in name and (target / name).resolve().is_relative_to(target.resolve())
        z.extractall(target)
    data_roots = [target / 'online']
assert len(data_roots) == 1, 'Attach exactly one original online corpus.'
data = data_roots[0]
env = os.environ | {'PYTHONPATH': str(code / 'src'), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'CUDA_VISIBLE_DEVICES': '', 'KAGGLE_API_TOKEN': token, 'PYTHONUNBUFFERED': '1'}
del token
print('Code and prepared data located. Checkpoints will be saved privately to:', CHECKPOINT_DATASET)
'''
runtime = '''# Isolated original runtime: Kaggle image updates cannot silently change training.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
install_env = os.environ | {'UV_PYTHON_INSTALL_DIR': '/tmp/h1-managed-python', 'UV_CACHE_DIR': '/tmp/h1-uv-cache'}
uv = [sys.executable, '-m', 'uv', '--no-config']
subprocess.run(uv + ['python', 'install', '3.12.13'], env=install_env, check=True)
runtime = Path('/tmp/h1-auto-runtime')
RUN_PYTHON = str(runtime / 'bin/python')
if not Path(RUN_PYTHON).exists():
    subprocess.run(uv + ['venv', '--seed', '--python', '3.12.13', str(runtime)], env=install_env, check=True)
subprocess.run(uv + ['pip', 'install', '--python', RUN_PYTHON, 'pip'], env=install_env, check=True)
subprocess.run(uv + ['pip', 'install', '--python', RUN_PYTHON, 'torch==2.10.0+cu128',
                    '--index-url', 'https://download.pytorch.org/whl/cu128'], env=install_env, check=True)
subprocess.run(uv + ['pip', 'install', '--python', RUN_PYTHON, 'numpy==2.0.2',
                    'pytest==8.4.2', 'datasketch==2.0.0', 'kaggle==2.2.4', 'kagglesdk==0.1.37'],
               env=install_env, check=True)
probe = "import platform,numpy,torch; assert (platform.python_version(),numpy.__version__,str(torch.__version__)) == ('3.12.13','2.0.2','2.10.0+cu128'); assert torch.cuda.device_count()==2; print('Pinned runtime ready:',[torch.cuda.get_device_name(i) for i in range(2)])"
subprocess.run([RUN_PYTHON, '-c', probe], env=env | {'CUDA_VISIBLE_DEVICES':'0,1'}, check=True)
'''
run = '''# One command: discover -> restore OR preflight -> backup -> train -> backup -> continue.
setup_hours = (time.monotonic() - setup_started) / 3600
minutes = min(SESSION_REMAINING_MINUTES, WEEKLY_GPU_HOURS_REMAINING * 60) - setup_hours * 60
assert minutes > 30, 'Not enough time remains for a safely backed-up chunk. Start a fresh session.'
command = [RUN_PYTHON, str(code / 'scripts/kaggle_auto.py'),
    '--inputs', str(inputs), '--data', str(data), '--config', str(code / 'configs/kaggle-stage1-paired.json'),
    '--tests', str(code / 'tests'), '--dataset', CHECKPOINT_DATASET,
    '--work', '/tmp/h1-automatic-work', '--output', '/kaggle/working/h1-automatic',
    '--minutes', str(minutes), '--weekly-hours', str(minutes / 60),
    '--prior-hours', str(EXTRA_UNRECORDED_HOURS + setup_hours), '--chunk-minutes', '30']
process = subprocess.Popen(command, env=env)
try:
    result = process.wait()
except KeyboardInterrupt:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        print('Driver is still stopping workers. Do not start another run; retain the last remote checkpoint.')
    raise
if result:
    raise RuntimeError('Automatic run stopped. Keep its output/error; do not switch to a new dataset or restart the experiment.')
print('Session finished. Remote backups are retained automatically. Use this same notebook and dataset next time.')
'''
cells = [cell('markdown', intro)]
for title, text in [('1. Settings and inputs', setup), ('2. Original runtime', runtime), ('3. Automatic run', run)]:
    cells += [cell('markdown', '## '+title+'\n'), cell('code', text)]
for i,c in enumerate(cells):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell-{i}', 'exec')
notebook = {'cells':cells, 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
            'nbformat':4, 'nbformat_minor':5}
for path in (root/'notebooks/kaggle-automatic.ipynb', out/'kaggle-automatic.ipynb'):
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
paths = list((root/'src').rglob('*.py')) + list((root/'tests').glob('test_*.py'))
paths += [root/'scripts/kaggle_auto.py', root/'pyproject.toml', root/'docs/kaggle-automatic.md']
paths += list((root/'configs').glob('*.json'))
with zipfile.ZipFile(out/'domain-shift-auto.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in paths:
        assert path.suffix != '.env' and path.name != '.env'
        z.write(path,path.relative_to(root).as_posix())
with zipfile.ZipFile(out/'domain-shift-auto.zip') as z:
    assert z.testzip() is None
    assert not any('.env' in Path(n).parts for n in z.namelist())
receipt = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
           if p.is_file() and p.name!='sha256.json'}
(out/'sha256.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Packaged:', out)
