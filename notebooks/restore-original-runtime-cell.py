# Run after notebook setup, before restoring the ORIGINAL local resume ZIP.
# This changes subprocess execution only, not the notebook kernel or frozen code.
from pathlib import Path
import json, os, signal, subprocess, sys, zipfile

resume_archive = Path('/kaggle/working/paired-resume-1790934440.zip')
assert resume_archive.is_file(), f'Keep the original archive available: {resume_archive}'
with zipfile.ZipFile(resume_archive) as z:
    saved = json.loads(z.read('freeze.json'))
    for condition in ('RMS', 'Taper-minus'):
        runtime = saved['runtimes'][condition]
        assert (runtime['python'], runtime['numpy'], runtime['torch']) == (
            '3.12.13', '2.0.2', '2.10.0+cu128'), 'This cell is for the reported runtime only.'

subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'uv'], check=True)
install_env = os.environ | {
    'UV_PYTHON_INSTALL_DIR': '/tmp/h1-managed-python',
    'UV_CACHE_DIR': '/tmp/h1-uv-cache',
}
uv = [sys.executable, '-m', 'uv', '--no-config']
subprocess.run(uv + ['python', 'install', '3.12.13'], env=install_env, check=True)
venv = Path('/tmp/h1-runtime-py31213')
RUN_PYTHON = str(venv / 'bin/python')
if not Path(RUN_PYTHON).exists():
    subprocess.run(uv + ['venv', '--python', '3.12.13', str(venv)], env=install_env, check=True)
subprocess.run(uv + ['pip', 'install', '--python', RUN_PYTHON,
    'torch==2.10.0+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128'],
    env=install_env, check=True)
subprocess.run(uv + ['pip', 'install', '--python', RUN_PYTHON,
    'numpy==2.0.2', 'pytest==8.4.2', 'datasketch==2.0.0'], env=install_env, check=True)

# Check the exact identity on both GPUs BEFORE changing the training launcher.
probe = r'''
import json, sys, zipfile, torch
from domain_shift_forgetting.pilot_runner import runtime_identity, source_identity
with zipfile.ZipFile(sys.argv[1]) as z:
    frozen = json.loads(z.read('workers/' + sys.argv[2] + '/freeze.json'))
expected = frozen['runtimes'][sys.argv[2]]
actual = runtime_identity(torch.device('cuda'))
differences = {k: {'saved': expected.get(k), 'current': actual.get(k)}
               for k in set(expected) | set(actual) if expected.get(k) != actual.get(k)}
assert frozen['code'] == source_identity(), 'Source code differs from the saved run.'
assert not differences, json.dumps(differences, indent=2)
print(sys.argv[2], 'original runtime matched:', actual, flush=True)
'''
for condition, gpu in [('RMS', '0'), ('Taper-minus', '1')]:
    subprocess.run([RUN_PYTHON, '-c', probe, str(resume_archive), condition],
                   env=env | {'CUDA_VISIBLE_DEVICES': gpu}, check=True)

def cli(module, *args):
    command = [RUN_PYTHON, '-m', 'domain_shift_forgetting.' + module, *map(str, args)]
    process = subprocess.Popen(command, env=env)
    try:
        result = process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            print('Coordinator did not close safely; preserve the prior archive.')
        raise
    if result:
        raise subprocess.CalledProcessError(result, command)

print('Runtime restored. Rerun the local restore cell using the ORIGINAL ZIP, then the training/export cell.')
print('Do not rerun the bootstrap cell afterward: it would replace this launcher.')
