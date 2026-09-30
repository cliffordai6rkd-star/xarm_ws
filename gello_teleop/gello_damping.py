"""Shared runtime damping configuration; never edits calibration mappings."""
from dataclasses import replace
from gello_teleop.uf_robot_gello_teleop import validate


def damping_config(robot, leader, side, block):
    if block is None:
        return leader
    if not isinstance(block, dict):
        raise ValueError('gello_damping must be a mapping')
    allowed = {'enabled', 'sample_rate_hz', 'sides', 'damping_mode',
               'damping_gain', 'damping_brake_gain', 'damping_current_limit',
               'damping_velocity_threshold', 'damping_velocity_filter_alpha',
               'damping_watchdog_ms', 'weak_hold_enabled', 'weak_hold_gain',
               'weak_hold_limit', 'weak_hold_release_velocity'}
    sides = block.get('sides', {})
    if set(block)-allowed or not isinstance(sides, dict) or set(sides)-{'left', 'right'}:
        raise ValueError('Invalid gello_damping fields or sides')
    override = sides.get(side, {})
    if not isinstance(override, dict) or set(override)-(allowed-{'sides'}):
        raise ValueError(f'Invalid gello_damping override for {side}')
    settings = {k: v for k, v in block.items() if k != 'sides'}
    settings.update(override)
    settings['damping_enabled'] = settings.pop('enabled', leader.damping_enabled)
    if 'sample_rate_hz' in settings:
        settings['fps'] = settings.pop('sample_rate_hz')
    leader = replace(leader, **settings)
    if type(leader.damping_enabled) is not bool or type(leader.weak_hold_enabled) is not bool:
        raise ValueError('GELLO damping enable flags must be booleans')
    if leader.damping_enabled and leader.damping_mode != 'current':
        raise ValueError('Enabled GELLO damping requires damping_mode: current')
    validate(robot, leader)
    return leader
