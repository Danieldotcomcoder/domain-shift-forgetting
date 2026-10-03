from pathlib import Path
import os, sys, subprocess, zipfile, shutil, tempfile

inputs = Path('/kaggle/input')
working = Path('/kaggle/working')
# Kaggle may expose the uploaded ZIP as an already-extracted directory.
roots = sorted({p.parent.parent.parent for p in inputs.rglob('kaggle.py')
                if p.parent.name == 'domain_shift_forgetting'
                and p.parent.parent.name == 'src'})
bundles = sorted(inputs.rglob('domain-shift-kaggle.zip'))
if roots:
    assert len(roots) == 1, f'Multiple code folders found: {roots}'
    code = roots[0]
elif bundles:
    assert len(bundles) == 1, f'Multiple code ZIPs found: {bundles}'
    code = Path(tempfile.mkdtemp(prefix='domain-shift-code-', dir=working))
    with zipfile.ZipFile(bundles[0]) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        z.extractall(code)
else:
    raise FileNotFoundError(
        f'No code bundle found under {inputs}. Attached files include: '
        f'{list(inputs.glob("*/*"))[:20]}')
assert (code / 'src/domain_shift_forgetting/kaggle.py').is_file(), code
print('Using code:', code)
env = os.environ | {'PYTHONPATH': str(code / 'src'), 'CUDA_VISIBLE_DEVICES': '0'}
# A new directory makes rerunning setup safe without overwriting earlier results.
results = Path(tempfile.mkdtemp(prefix='kaggle-results-', dir=working))
(results / 'pip-freeze.txt').write_text(subprocess.check_output(
    [sys.executable, '-m', 'pip', 'freeze'], text=True))
(results / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
print((results / 'nvidia-smi.txt').read_text())
print('Results directory:', results)
