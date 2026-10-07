import numpy as np
import pytest
from collections import deque
from types import SimpleNamespace
import threading

from inference.dp_wm.core import HistorySnapshot
from inference.dp_wm.live import LiveSource, _cadence_samples
from inference.pi0_wm.wm import CausalChain


def test_one_delayed_state_is_interpolated_from_real_endpoints_at_valid_cadence():
    before={key:np.zeros(7) for key in ('q','dq','tau','delta_q')}
    after={key:np.ones(7) for key in before}
    samples=_cadence_samples(1_000_000,before,1_020_000,after)
    assert len(samples)==2
    assert [t for t,_ in samples]==[1_010_000,1_020_000]
    np.testing.assert_allclose(samples[0][1]['q'],.5)
    stamps=1_000_000+np.arange(48,dtype=np.int64)*10_000
    stamps=np.r_[stamps,samples[0][0]+470_000,samples[1][0]+470_000]
    history=HistorySnapshot(stamps,{key:np.zeros((50,7)) for key in before})
    history.validate()


def test_early_duplicate_is_skipped_and_long_gap_requires_history_rebuild():
    raw={key:np.zeros(7) for key in ('q','dq','tau','delta_q')}
    assert _cadence_samples(1_000_000,raw,1_004_000,raw)==[]
    with pytest.raises(ValueError,match='bridge limit'):
        _cadence_samples(1_000_000,raw,1_025_000,raw)


@pytest.mark.parametrize('strict', [False, True])
def test_live_reader_preserves_measured_rows_or_explicit_legacy_bridge(strict):
    source=LiveSource.__new__(LiveSource)
    source.stop_event=threading.Event()
    source.lock=threading.Lock()
    source.states=deque(maxlen=100)
    source.rows=deque(maxlen=50)
    source.options={'command_source':'controller_reference'}
    source.strict_measured_history=strict
    source.operations={}
    source.filters={key:CausalChain([]) for key in ('q','dq','tau','delta_q')}
    source.history_interpolated_rows=0
    source.history_rebuilds=0
    source.last_history_gap_ms=None
    source.status='waiting'
    source.error=None
    source._command=lambda stamp:(np.ones(7),stamp)
    class Arm:
        stamps=iter((1_000_000,1_010_000,1_030_000,1_040_000))
        def read_joint_state(self):
            stamp=next(self.stamps)
            if stamp==1_040_000:
                source.stop_event.set()
            return SimpleNamespace(q_timestamp_us=stamp,timestamp_us=stamp,q_valid=True,
                dq_valid=True,torque_valid=True,q=np.full(7,(stamp-1_000_000)/10_000),
                dq=np.zeros(7),torque=np.zeros(7))
    source.arm=Arm()
    source._read()
    assert source.error is None
    if strict:
        assert [t for t,_ in source.rows] == [1_040_000]
        assert source.history_interpolated_rows == 0
        assert source.history_rebuilds == 1
        return
    assert [t for t,_ in source.rows]==[1_000_000,1_010_000,1_020_000,1_030_000,1_040_000]
    assert source.history_interpolated_rows==1
    assert source.history_rebuilds==0
    np.testing.assert_allclose(source.rows[2][1]['q'],2.)
