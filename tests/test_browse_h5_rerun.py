from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import browse_h5_rerun as browser


def test_discovery_supports_both_collectors_and_rejects_ambiguous_indices(tmp_path):
    old = tmp_path / 'episode_0002_old.h5'
    new = tmp_path / 'dual_gello_xarm_0047_new.h5'
    for path in (old, new, tmp_path / 'unrelated.h5'):
        path.touch()
    assert browser.discover_episodes(tmp_path) == {2: old, 47: new}
    (tmp_path / 'dual_gello_xarm_0002_duplicate.h5').touch()
    with pytest.raises(RuntimeError, match='multiple files'):
        browser.discover_episodes(tmp_path)


def test_browser_forwards_arm_and_estimation_and_preserves_trash(tmp_path, monkeypatch):
    folder = tmp_path / 'runs' / 'dataset'
    folder.mkdir(parents=True)
    first = folder / 'dual_gello_xarm_0000_a.h5'
    second = folder / 'dual_gello_xarm_0001_b.h5'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    trash = folder.parent / 'trash'
    trash.mkdir()
    (trash / second.name).write_bytes(b'previous')
    calls = []
    monkeypatch.setattr(browser, 'run_rerun', lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    actions = iter(('0', '1', 'd', 'q'))
    assert browser.browse(folder, 'left_wrist/right_wrist',
                          input_fn=lambda _: next(actions), output_fn=lambda _: None,
                          arm='right', estimate_tau_ext=True) == 0
    assert [args[1] for args, _ in calls] == [0, 1]
    assert all(kwargs == {'arm': 'right', 'estimate_tau_ext': True} for _, kwargs in calls)
    assert first.read_bytes() == b'first'
    assert not second.exists()
    assert (trash / second.name).read_bytes() == b'previous'
    assert (trash / (second.stem + '_1.h5')).read_bytes() == b'second'


def test_renderer_command_uses_current_python_and_all_options(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(browser.subprocess, 'run',
                        lambda cmd, **kwargs: commands.append((cmd, kwargs)) or SimpleNamespace(returncode=3))
    assert browser.run_rerun(tmp_path, 4, 'right_wrist', arm='left', estimate_tau_ext=False) == 3
    command, kwargs = commands[0]
    assert command[0] == browser.sys.executable
    assert Path(command[1]).is_file()
    assert command[2:] == [str(tmp_path), '--episode', '4', '--arm', 'left',
                           '--camera', 'right_wrist', '--no-estimate-tau-ext']
    assert kwargs == {'check': False}
