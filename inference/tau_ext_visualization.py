"""Read-only live external torque preview using the collection estimator."""
from __future__ import annotations

from types import SimpleNamespace
import logging

from gello_teleop.tau_free_feedback import TauFreeFeedbackWorker
from xarm_stack.torque_visualization import TorqueVisualizer

log = logging.getLogger(__name__)


class TauExtPreview:
    """Estimate on a bounded worker queue; never issue robot or GELLO commands."""

    def __init__(self, options, hardware, config_path, side, arm):
        settings = dict(hardware.get('force_feedback', {}))
        settings['enabled'] = True  # Computation only, independent of force feedback enablement.
        if settings.get('source', 'checkpoint') != 'urdf':
            raise ValueError('tau_ext visualization requires force_feedback.source: urdf')
        self.side = side
        self.worker = TauFreeFeedbackWorker(settings, config_path, (side,), 100.)
        self.worker.bind_payload_source(side, arm)
        self.stale_s = self.worker.stale_s
        self.method = self.worker.settings[side]['dynamics_method']
        self.label = f'URDF {self.method}'
        self.plot = None
        if options['plot']:
            plot = dict(hardware.get('torque_visualization', {}))
            plot.update(enabled=True, force_feedback_source='urdf', learned_torque={'enabled': False})
            # Use precisely the estimator URDF, gravity and payload on the plot path.
            plot_settings = dict(self.worker.settings[side])
            plot_settings['payload'] = plot_settings.get('payload') or {}
            plot['sides'] = {side: plot_settings}
            self.plot = TorqueVisualizer.from_config(plot, config_path, (side,))
            self.plot.bind_payload_source(side, arm)

    def start(self):
        self.worker.start()
        if self.plot is not None:
            try:
                self.plot.start()
            except Exception:
                log.warning('tau_ext time-series window unavailable; other previews continue', exc_info=True)
                self.plot.close()
                self.plot = None

    def submit(self, state):
        # URDF estimation consumes q/dq/tau (and ddq for RNEA). The empty
        # command field makes explicit that no host command is invented here.
        sample = SimpleNamespace(timestamp_us=state.timestamp_us, follower_states=(state,),
                                 q_cmd=[], q_cmd_ok=(True,))
        self.worker.submit(sample, active=True)
        if self.plot is not None:
            self.plot.publish((self.side,), (state,), urdf_results=(self.snapshot(),))

    def snapshot(self):
        return self.worker.snapshot(0)

    def reset(self):
        self.worker.invalidate()

    def close(self):
        try:
            self.worker.close()
        finally:
            if self.plot is not None:
                self.plot.close()
