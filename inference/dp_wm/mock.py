"""Explicit synthetic data/models for scheduler checks, no device access."""
from types import SimpleNamespace
import numpy as np
from inference.core.contracts import ActionChunk
from inference.dp_wm.core import HistorySnapshot
from nero_collection.time_utils import now_us


class MockSource:
    status = 'synthetic inputs (not real inference)'

    def __init__(self, image_keys=('front', 'left_wrist', 'right_wrist')):
        self.image_keys = image_keys

    def start(self):
        pass

    def close(self):
        pass

    def control_state(self):
        from inference.controllers.wm_two_step import ControlState

        return ControlState(now_us() // 10_000 * 10_000, np.zeros(7), np.zeros(7), np.zeros(7))

    def poll(self):
        anchor = now_us() // 10_000 * 10_000
        image_anchor = anchor // 40_000 * 40_000
        observations = tuple(SimpleNamespace(timestamp_us=t, q=np.zeros(7),
            images={key: np.zeros((224,224,3), np.uint8) for key in self.image_keys},
            metadata={}) for t in (image_anchor-40_000, image_anchor))
        history = HistorySnapshot(anchor + np.arange(-49,1,dtype=np.int64)*10_000,
            {key: np.zeros((50,7), np.float32) for key in ('q','dq','delta_q','tau')})
        return observations, history


class MockPolicy:
    def predict_observations(self, observations, *, plan_steps):
        return ActionChunk(np.zeros((plan_steps,7)), 'joint', None,
                           observations[-1].timestamp_us, .04)


class MockWM:
    def infer(self, request):
        history = request['history']
        return {'q': np.zeros((1,32,7)), 'tau': np.zeros((1,32,7)),
                'contact_probability': np.tile([1.,0.,0.], (1,32,1)),
                'anchor_us': history.anchor_us, 'plan_version': request['plan_version'],
                'anchor_tau_filtered': history.values['tau'][-1].copy(),
                'deadline_us': request['deadline_us'],
                'future_timestamps_us': history.anchor_us + np.arange(1,33,dtype=np.int64)*10_000}
