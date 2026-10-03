"""Cloud round-trip / orchestration tests. No Kaggle credentials or uploads used."""
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from domain_shift_forgetting import paired, pilot_control as pc
from domain_shift_forgetting.pilot_runner import source_identity

spec = importlib.util.spec_from_file_location("auto_driver", Path(__file__).parents[1] / "scripts/kaggle_auto.py")
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


class FakeAPI:
    def __init__(self, cloud):
        self.cloud = cloud
        self.version = 0
        self.private = None
        self.preserve_history = None
        self.corrupt = False

    def copy(self, folder):
        self.version += 1
        shutil.copytree(folder, self.cloud / str(self.version))
        return SimpleNamespace(error=None)

    def dataset_create_new(self, folder, **kw):
        self.private = not kw['public']
        return self.copy(folder)

    def dataset_create_version(self, folder, notes, **kw):
        self.preserve_history = not kw['delete_old_versions']
        return self.copy(folder)

    def dataset_download_file(self, dataset, name, path, **kw):
        shutil.copyfile(self.cloud / dataset.split('/')[-1] / name, Path(path) / name)
        if self.corrupt and name.endswith('.bin'):
            with (Path(path) / name).open('ab') as f:
                f.write(b'corrupt')


class Store(auto.KaggleStore):
    def info(self):
        return self.api.version or None


def fixture(root, *, admitted=True):
    pc.write_json(root / 'freeze.json', {'layout':'paired-v1', 'admitted':admitted})
    for condition in ('RMS', 'Taper-minus'):
        child = root / 'workers' / condition
        pc.write_json(child / 'freeze.json', {'layout':'paired-v1', 'admitted':admitted})
        pc.write_json(child / 'status.json', {'status':'paused'})
        run = child / 'runs' / ('S101-' + condition)
        run.mkdir(parents=True)
        generation = run / '.switch-generation.pt'
        generation.write_bytes(b'synthetic full checkpoint')
        pc.write_json(run / 'switch.pt.json', {'file':generation.name, 'sha256':auto.sha(generation), 'previous':None})
        (run / 'weights-prefix-1.pt').write_bytes(b'historical weights')


def test_private_cloud_roundtrip_registers_both_prefixes_and_offloads_history(tmp_path):
    root = tmp_path / 'run'
    fixture(root)
    archive = tmp_path / 'saved.zip'
    auto.archive_clean(root, archive)
    api = FakeAPI(tmp_path / 'cloud')
    store = Store('user/h1-checkpoints', tmp_path / 'cache', api=api)
    head = store.publish({'phase':'initial'}, expected=None)
    head = store.publish(head, archive=archive, expected=head['version'])
    downloaded, proof = store.fetch_checkpoint(head, tmp_path / 'download')
    original_guard = pc.require_persisted_input
    auto.import_verified(downloaded, tmp_path / 'restored', proof)
    assert pc.require_persisted_input is original_guard
    assert api.private and api.preserve_history
    for condition in ('RMS', 'Taper-minus'):
        child = tmp_path / 'restored/workers' / condition
        assert len(auto.read(child / 'durable-prefixes.json')['prefixes']) == 1
        assert len(auto.read(child / 'archived-weights.json')) == 1
        assert not list(child.rglob('weights-*.pt'))
    assert store.head()['checkpoint']['sha256'] == auto.sha(archive)


def test_corrupt_download_cannot_acknowledge_backup(tmp_path):
    archive = tmp_path / 'saved.zip'
    fixture(tmp_path / 'run')
    auto.archive_clean(tmp_path / 'run', archive)
    api = FakeAPI(tmp_path / 'cloud')
    store = Store('user/h1-checkpoints', tmp_path / 'cache', api=api)
    head = store.publish({'phase':'training'}, archive=archive)
    api.corrupt = True
    with pytest.raises(ValueError, match='checksum'):
        store.fetch_checkpoint(head, tmp_path / 'download')
    assert archive.exists() and (tmp_path / 'run').exists()


def test_unverified_local_archive_remains_rejected(tmp_path):
    archive = tmp_path / 'fake.zip'
    archive.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='verification'):
        auto.import_verified(archive, tmp_path / 'restored', {})
    assert not (tmp_path / 'restored').exists()


def test_failed_import_restores_original_source_guard(tmp_path):
    archive = tmp_path / 'fake.zip'
    archive.write_bytes(b'not a zip')
    proof = {'schema':'kaggle-readback-v1','dataset':'user/h1','version':1,'sha256':auto.sha(archive),'verified':True}
    guard = pc.require_persisted_input
    with pytest.raises(Exception):
        auto.import_verified(archive, tmp_path / 'restored', proof)
    assert pc.require_persisted_input is guard
    assert (tmp_path / 'restored/IMPORT-FAILED.json').exists()


def test_store_conflict_prevents_overwrite(tmp_path):
    api = FakeAPI(tmp_path / 'cloud')
    store = Store('user/h1-checkpoints', tmp_path / 'cache', api=api)
    store.publish({'phase':'initial'})
    with pytest.raises(RuntimeError, match='changed'):
        store.publish({'phase':'initial'}, expected=None)
    assert api.version == 1


def test_guarded_cleanup_and_failed_state(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        auto.safe_remove(tmp_path, tmp_path)
    fixture(tmp_path / 'failed')
    pc.write_json(tmp_path / 'failed/failure.json', {'error':'fixture'})
    with pytest.raises(ValueError, match='Failed run'):
        auto.archive_clean(tmp_path / 'failed', tmp_path / 'archive.zip')


def test_preflight_expected_failed_import_fixture_is_not_a_run_failure(tmp_path):
    root=tmp_path/'preflight'
    fixture(root)
    pc.write_json(root/'workers/RMS/test-tmp/test_reports_archive/bad-restore/IMPORT-FAILED.json',{'status':'failed'})
    pc.write_json(root/'workers/Taper-minus/test-tmp/test_tampering/bad-restore/IMPORT-FAILED.json',{'status':'failed'})
    auto.archive_clean(root,tmp_path/'valid-preflight.zip')
    import zipfile
    with zipfile.ZipFile(tmp_path/'valid-preflight.zip') as z:
        assert not any('test-tmp' in name for name in z.namelist())
    pc.write_json(root/'workers/RMS/IMPORT-FAILED.json',{'status':'failed'})
    with pytest.raises(ValueError,match='Failed run'):
        auto.archive_clean(root,tmp_path/'must-not-exist.zip')


def test_discovery_chooses_latest_same_experiment_and_rejects_ambiguity(tmp_path):
    for index in (1,2):
        root = tmp_path / f'output{index}'
        fixture(root)
        pc.write_json(root / 'notebook-ledger.json', {'external_gpu_hours':1,'sessions':[
            {'started_utc':f'2026-10-03T0{index}:00:00Z','charged_seconds':60,'status':'paused'}]})
        pc.export_run(root, tmp_path / f'paired-resume-{index}.zip')
        shutil.rmtree(root)
    assert auto.discover(tmp_path).name == 'paired-resume-2.zip'
    root = tmp_path / 'other'
    fixture(root, admitted=False)
    pc.write_json(root / 'notebook-ledger.json', {'sessions':[]})
    pc.export_run(root, tmp_path / 'paired-resume-3.zip')
    shutil.rmtree(root)
    with pytest.raises(ValueError, match='different experiments'):
        auto.discover(tmp_path)


def test_automatic_preflight_start_prefix_backup_and_next_session_resume(tmp_path, monkeypatch):
    project = Path(__file__).parents[1]
    api = FakeAPI(tmp_path / 'cloud')
    monkeypatch.setattr(auto, 'KaggleStore', lambda dataset, cache: Store(dataset, cache, api=api))
    data = tmp_path / 'inputs/online'
    pc.write_json(data / 'manifest.json', {'fixture':1})
    pc.write_json(data.parent / 'orders/manifest.json', {'fixture':1})
    args = SimpleNamespace(inputs=tmp_path/'inputs', data=data, orders=data.parent/'orders',
        config=project/'configs/kaggle-stage1-paired.json', tests=project/'tests',
        work=tmp_path/'work', output=tmp_path/'output', dataset='user/h1-checkpoints',
        minutes=300, weekly_hours=10, prior_hours=0, chunk_minutes=30)
    calls = []
    def preflight(a):
        calls.append('preflight')
        for c in ('RMS','Taper-minus'):
            pc.write_json(a.root/'workers'/c/'validation.json', {'status':'passed'})
        pc.write_json(a.root/'paired-preflight.json', {'status':'passed'})
    def start(a):
        calls.append('start')
        assert '--verify' in paired.command('_verify', root=a.preflight/'workers/RMS', data=a.data,
            orders=a.orders, output=a.root/'proof.json')
        assert (a.preflight/'workers/RMS/automatic-remote-proof.json').exists()
        fixture(a.root)
    def run(a):
        calls.append('run')
        for c in ('RMS','Taper-minus'):
            durable = auto.read(a.root/'workers'/c/'durable-prefixes.json')
            assert durable['prefixes']  # Verified cloud backup clears prefix stop automatically.
        pc.write_json(a.root/'status.json', {'status':'completed' if calls.count('run')==2 else 'paused'})
        return {'decision': {'category':'fixture'}}
    monkeypatch.setattr(paired, 'preflight', preflight)
    monkeypatch.setattr(paired, 'start', start)
    monkeypatch.setattr(paired, 'run', run)
    auto.run_auto(args)
    assert calls == ['preflight','start','run','run']
    assert len(list(args.work.glob('download-*'))) == 1
    assert not list(args.work.glob('pending-*.zip'))
    assert len(list(args.work.glob('restored-*'))) == 1
    # New session, no manually attached checkpoint: detect remote completion, never restart.
    args.work = tmp_path/'next-session'
    auto.run_auto(args)
    assert calls == ['preflight','start','run','run']
    assert api.private and api.preserve_history


def test_scientific_source_bundle_is_unchanged():
    # Compare against the actual paired preflight code used by the recovered run.
    import zipfile
    p = Path(__file__).parents[1]/'artifacts/kaggle-paired/domain-shift-paired.zip'
    with zipfile.ZipFile(p) as z:
        for name, digest in source_identity().items():
            assert auto.hashlib.sha256(z.read('src/domain_shift_forgetting/'+name)).hexdigest() == digest


def test_download_large_file_wrapper(tmp_path):
    import zipfile
    class WrappedAPI:
        def dataset_download_file(self, dataset, name, path, **kwargs):
            with zipfile.ZipFile(Path(path)/(name+'.zip'),'w') as z:
                z.writestr(name, b'opaque checkpoint payload')
    store = Store('user/h1-checkpoints', tmp_path/'cache', api=WrappedAPI())
    path = store.download(1, 'checkpoint.bin', tmp_path/'download')
    assert path.read_bytes() == b'opaque checkpoint payload'
    assert not path.with_suffix('.bin.zip').exists()


def test_download_wrapper_refuses_path_traversal(tmp_path):
    import zipfile
    class WrappedAPI:
        def dataset_download_file(self, dataset, name, path, **kwargs):
            with zipfile.ZipFile(Path(path)/(name+'.zip'),'w') as z:
                z.writestr('../outside', b'bad')
    store = Store('user/h1-checkpoints', tmp_path/'cache', api=WrappedAPI())
    with pytest.raises(ValueError, match='wrapper'):
        store.download(1,'checkpoint.bin', tmp_path/'download')


def test_adopted_snapshot_keeps_original_weight_history(tmp_path, monkeypatch):
    import zipfile
    project = Path(__file__).parents[1]
    api = FakeAPI(tmp_path/'cloud')
    monkeypatch.setattr(auto,'KaggleStore',lambda dataset,cache: Store(dataset,cache,api=api))
    recovered = tmp_path/'inputs/recovered'
    fixture(recovered)
    frozen = {'layout':'paired-v1','admitted':True,'code':source_identity(),
              'policy':pc.load_policy(project/'configs/kaggle-stage1-paired.json')}
    pc.write_json(recovered/'freeze.json', frozen)
    pc.write_json(recovered/'notebook-ledger.json',{'external_gpu_hours':10,'sessions':[
        {'started_utc':'2026-10-03T00:00:00Z','charged_seconds':3600,'status':'paused'}]})
    pc.export_run(recovered,tmp_path/'inputs/paired-resume-1790983719.zip')
    shutil.rmtree(recovered)
    data = tmp_path/'inputs/online'
    pc.write_json(data/'manifest.json',{'fixture':1})
    pc.write_json(data.parent/'orders/manifest.json',{'fixture':1})
    args=SimpleNamespace(inputs=tmp_path/'inputs',data=data,orders=data.parent/'orders',
        config=project/'configs/kaggle-stage1-paired.json',tests=project/'tests',work=tmp_path/'work',
        output=tmp_path/'output',dataset='user/h1-checkpoints',minutes=25,weekly_hours=10,prior_hours=0,chunk_minutes=30)
    monkeypatch.setattr(paired,'preflight',lambda *a: pytest.fail('Adoption must not repeat preflight'))
    monkeypatch.setattr(paired,'start',lambda *a: pytest.fail('Adoption must not restart'))
    auto.run_auto(args)
    store=Store(args.dataset,tmp_path/'inspect',api=api)
    head=store.head()
    p,_=store.fetch_checkpoint(head,tmp_path/'verify')
    with zipfile.ZipFile(p) as z:
        assert len([n for n in z.namelist() if 'weights-prefix-' in n])==2
    assert head['budget_hours'] >= 11


def test_private_missing_403_requires_authenticated_owned_inventory(tmp_path):
    # Exact SDK interface, but all network calls are mocked.
    pytest.importorskip('kagglesdk')
    import requests
    from contextlib import contextmanager
    response=requests.Response();response.status_code=403
    class API:
        @contextmanager
        def build_kaggle_client(self):
            def denied(*a):raise requests.HTTPError(response=response)
            yield SimpleNamespace(datasets=SimpleNamespace(dataset_api_client=SimpleNamespace(get_dataset=denied)))
        def dataset_list(self, mine, page):
            return [SimpleNamespace(ref='user/existing-owned-data')] if page==1 else []
    store=auto.KaggleStore('user/new-checkpoints',tmp_path/'cache',api=API())
    assert store.info() is None
    store.api.dataset_list=lambda **kw: []
    with pytest.raises(RuntimeError,match='Cannot read'):
        store.info()
    store.api.dataset_list=lambda **kw:[SimpleNamespace(ref='user/new-checkpoints')]
    with pytest.raises(RuntimeError,match='exists but access'):
        store.info()


def test_working_output_staging_does_not_claim_external_backup(tmp_path, monkeypatch):
    root=tmp_path/'original'
    fixture(root)
    pc.write_json(root/'notebook-ledger.json',{'sessions':[]})
    working=tmp_path/'working';working.mkdir()
    archive=working/'paired-resume-1.zip'
    pc.export_run(root,archive)
    assert auto.discover(tmp_path/'inputs', working)==archive
    monkeypatch.setattr(pc,'KAGGLE_WORKING',working)
    monkeypatch.setattr(pc,'KAGGLE_INPUT',tmp_path/'inputs')
    with pytest.raises(ValueError,match='persisted archive'):
        pc.import_run(archive,tmp_path/'not-allowed')
    guard=pc.require_persisted_input
    auto.stage_candidate(archive,tmp_path/'staged')
    assert pc.require_persisted_input is guard
    for receipt in (tmp_path/'staged').rglob('durable-prefixes.json'):
        assert not auto.read(receipt)['prefixes']
