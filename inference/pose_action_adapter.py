"""Convert absolute base-frame xyz/quaternion chunks to xArm joint targets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from inference.policies.errors import JointActionContractError


class PoseActionAdapter:
    def __init__(self, settings, interface, *, joint_limits=None, model=None):
        self.settings = dict(settings)
        self.target_frame = interface['coordinate_frame']
        self.quaternion_order = interface['quaternion_order']
        self.frame_name = self.settings.get('frame_name', self.target_frame)
        allowed = {'urdf_path', 'frame_name', 'frame_offset_xyz_rpy', 'max_iterations',
                   'position_tolerance_m', 'rotation_tolerance_rad', 'damping', 'step_gain',
                   'maximum_iteration_step_rad', 'joint_position_margin_rad', 'quaternion_tolerance'}
        if extra := set(self.settings) - allowed:
            raise ValueError(f'Unknown ik settings: {sorted(extra)}')
        self.max_iterations = self.settings.get('max_iterations', 100)
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError('ik.max_iterations must be a positive integer')
        for key, default in {'position_tolerance_m': 1e-4, 'rotation_tolerance_rad': 1e-3,
                             'damping': 1e-3, 'step_gain': 1., 'maximum_iteration_step_rad': .2,
                             'joint_position_margin_rad': .0, 'quaternion_tolerance': .15}.items():
            value = self.settings.get(key, default)
            if isinstance(value, bool):
                raise ValueError(f'ik.{key} must be numeric')
            value = float(value)
            if not np.isfinite(value) or (value < 0 if key == 'joint_position_margin_rad' else value <= 0):
                raise ValueError(f'Invalid ik.{key}')
            setattr(self, key, value)
        if self.quaternion_tolerance >= 1:
            raise ValueError('ik.quaternion_tolerance must be less than one')
        if interface.get('reference_frame', 'base') not in {'base', 'link_base'}:
            raise ValueError('EEP pose IK requires absolute poses in the robot base frame')
        offset = self.settings.get('frame_offset_xyz_rpy')
        if self.frame_name != self.target_frame and offset is None:
            raise ValueError('ik.frame_name differs from coordinate_frame: configure frame_offset_xyz_rpy')
        self.frame_offset = np.eye(4)
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            if offset.shape != (6,) or not np.isfinite(offset).all():
                raise ValueError('ik.frame_offset_xyz_rpy must be [x,y,z,roll,pitch,yaw] in m/rad')
            self.frame_offset[:3, 3] = offset[:3]
            self.frame_offset[:3, :3] = Rotation.from_euler('xyz', offset[3:]).as_matrix()
        self.inverse_offset = np.linalg.inv(self.frame_offset)
        if model is None:
            from nero_collection.control.dynamics import PinocchioDynamicsModel

            path = self.settings.get('urdf_path')
            if not path or not Path(path).is_file():
                raise ValueError('EEP pose IK requires a valid ik.urdf_path')
            model = PinocchioDynamicsModel(path, frame_name=self.frame_name, locked_joint_names=())
            if tuple(model.model.names[1:]) != tuple(f'joint{i}' for i in range(1, 8)):
                raise ValueError('IK URDF must use xArm joint1..joint7 order')
        self.model = model
        bounds = np.column_stack([model.position_lower, model.position_upper]).astype(float)
        if joint_limits is not None:
            bounds[:, 0] = np.maximum(bounds[:, 0], joint_limits[:, 0])
            bounds[:, 1] = np.minimum(bounds[:, 1], joint_limits[:, 1])
        self.lower = bounds[:, 0] + self.joint_position_margin_rad
        self.upper = bounds[:, 1] - self.joint_position_margin_rad
        if (bounds.shape != (7, 2) or not np.isfinite(bounds).all()
                or np.any(self.lower >= self.upper)):
            raise ValueError('Invalid or empty IK joint limits')

    def forward_pose(self, q):
        """Base-to-target pose, also usable for a reachable offline mock target."""
        pose = self.model.snapshot(np.asarray(q, dtype=float), np.zeros(7)).pose @ self.frame_offset
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        if self.quaternion_order == 'wxyz':
            quat = quat[[3, 0, 1, 2]]
        return np.r_[pose[:3, 3], quat]

    def convert(self, actions, q_seed):
        values = np.asarray(actions, dtype=float)
        q = np.asarray(q_seed, dtype=float).copy()
        if (values.ndim != 2 or values.shape[1] != 7 or not len(values)
                or not np.isfinite(values).all() or q.shape != (7,) or not np.isfinite(q).all()):
            raise JointActionContractError('IK needs finite [H,7] poses and seven measured seed joints')
        norms = np.linalg.norm(values[:, 3:], axis=1)
        if np.any(norms < 1e-8) or np.any(np.abs(norms - 1.) > self.quaternion_tolerance):
            raise JointActionContractError('EEP pose quaternion is zero or outside unit-norm tolerance')
        output = []
        diagnostics = []
        for index, action in enumerate(values):
            quat = action[3:] / norms[index]
            if self.quaternion_order == 'wxyz':
                quat = quat[[1, 2, 3, 0]]
            target = np.eye(4)
            target[:3, 3] = action[:3]
            target[:3, :3] = Rotation.from_quat(quat).as_matrix()
            # target = base->action frame; the model solves base->URDF frame.
            target = target @ self.inverse_offset
            try:
                q, diagnostic = self.solve(q, target)
            except JointActionContractError as exc:
                raise JointActionContractError(f'EEP pose action step {index}: {exc}') from exc
            output.append(q.copy())
            diagnostics.append(diagnostic)
        return np.stack(output), diagnostics

    def solve(self, q_seed, target):
        q = np.clip(q_seed, self.lower, self.upper)
        for iteration in range(self.max_iterations + 1):
            snapshot = self.model.snapshot(q, np.zeros(7))
            position = target[:3, 3] - snapshot.pose[:3, 3]
            rotation = Rotation.from_matrix(target[:3, :3] @ snapshot.pose[:3, :3].T).as_rotvec()
            position_error = float(np.linalg.norm(position))
            rotation_error = float(np.linalg.norm(rotation))
            if position_error <= self.position_tolerance_m and rotation_error <= self.rotation_tolerance_rad:
                return q.copy(), {'iterations': iteration, 'position_error_m': position_error,
                                  'rotation_error_rad': rotation_error}
            if iteration == self.max_iterations:
                break
            jacobian = np.asarray(snapshot.jacobian, dtype=float)
            if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
                raise JointActionContractError('IK requires a finite base-aligned 6x7 Jacobian')
            error = np.r_[position, rotation]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*self.damping**2, error)
            delta *= self.step_gain
            length = np.linalg.norm(delta)
            if length > self.maximum_iteration_step_rad:
                delta *= self.maximum_iteration_step_rad / length
            q = np.clip(q + delta, self.lower, self.upper)
        raise JointActionContractError(f'IK failed for frame {self.target_frame}: '
                                       f'position_error={position_error:.6f} m, '
                                       f'rotation_error={rotation_error:.6f} rad; no action chunk returned')
