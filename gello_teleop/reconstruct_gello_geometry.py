#!/usr/bin/env python3
"""Reconstruct an explicitly provisional GELLO assembly from STL mounting datums.

No CAD, hardware access, or additional Python packages are needed. Datums below
were recovered by cylindrical surface fits; the companion JSON records the fits.
The horn clocking, motor seating and shaft-end choices remain assembly hypotheses.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gello_teleop.build_gello_urdf import build_model, template, write_xml


def frame(normal, guide):
    z = np.asarray(normal, float)
    z /= np.linalg.norm(z)
    x = np.asarray(guide, float)
    x -= z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def rpy(rotation):
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(np.cos(pitch)) > 1e-8:
        roll, yaw = np.arctan2(rotation[2, 1], rotation[2, 2]), np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll, yaw = np.arctan2(-rotation[1, 2], rotation[1, 1]), 0.0
    return [float(roll), float(pitch), float(yaw)]


# Millimetres in each ORIGINAL STL coordinate system. Incoming points lie on
# the flat horn mating surface, not the bounding-box centre or screw mid-depth.
# Each tuple is (point, material-facing normal, chosen horn clocking guide).
HORNS = [
    ([0, -6, 2.5], [0, 0, 1], [1, 0, 0]),
    ([0, 26.5, -89], [0, 1, 0], [1, 0, 0]),
    ([0, 0, 2.5], [0, 0, 1], [1, 0, 0]),
    ([-12.5, 10, 0], [1, 0, 0], [0, 0, 1]),
    ([1, 83.75, 0], [0, -1, 0], [1, 0, 0]),
    ([2.5, 15, -15], [1, 0, 0], [0, 0, 1]),
    ([-15.27189-1.45*0.052336, -19.49378-1.45*0.99863, 0],
     [0.052336, 0.99863, 0], [0, 0, 1]),
]
# Rear-case seating face, long direction TOWARDS output shaft, output normal,
# clocking guide, case width. Body depth 23 + horn protrusion 3 = 26 mm.
# In particular a 16x30 mm bolt rectangle has the shaft 7.5 mm off its centre.
CASES = [
    ([-62.5, 10, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]),
    ([0, 4.8, 48.5], [0, 0, -1], [0, 1, 0], [1, 0, 0], [1, 0, 0]),
    ([0, 17.5, -9.7], [0, -1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0]),
    ([18.4, 15.002275, 37.630755], [0, -0.477159, -0.878817], [1, 0, 0],
     [0, 0.878817, -0.477159], [0, 0.878817, -0.477159]),
    ([-48.5, 69.916855+1.15*0.731354, -26.048045-1.15*0.681998],
     [0, -0.681998, -0.731354], [0, 0.731354, -0.681998], [1, 0, 0], [1, 0, 0]),
    ([19.7, 7.5, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 1]),
    ([2.5, -16.3, 15.82438], [0, 0, -1], [0, -1, 0], [1, 0, 0], [1, 0, 0]),
]

# Coordinate convention for the bent reference [0,0,0,90,0,90,0] degrees:
# shoulder/elbow/wrist pitch axes lie parallel, forearm roll is horizontal and
# the handle output points down. Clocking follows the corrected mounting sides,
# including L5's opposite horn face, and the two tilted STL mounting frames.
# It is a virtual coordinate convention, not a measured photo angle.
# This defines a better virtual datum; it does NOT certify real encoder offsets.
REFERENCE_CLOCK_DEGREES = [0., 0., -90., 71.5, 0., 180., 180.]
DISPLAY_MIDDLE_Q_DEG = [90., 0., -90., 90., 0., 90., -90.]


def reconstruct(clock_degrees=None):
    clock = np.asarray(REFERENCE_CLOCK_DEGREES if clock_degrees is None else clock_degrees, float)
    if clock.shape != (7,) or not np.isfinite(clock).all():
        raise ValueError('Supply seven finite horn clocking angles')
    config = template()
    config.update(mesh_units_verified=True, geometry_verified=False,
                  geometry_source='STL mounting-hole fits + ROBOTIS X330 drawing and nominal STEP geometry',
                  assembly_status='mounting sides corrected after overlap audit; physical review still pending',
                  joint_zero_status='virtual assembly datum, NOT the existing teleoperation encoder zero',
                  horn_clock_degrees=clock.tolist(),
                  comparison_reference_q_deg=DISPLAY_MIDDLE_Q_DEG.copy(),
                  comparison_reference_source='User-selected viewer middle pose; not an encoder calibration',
                  note='Offline reconstruction. No CAD needed. Lengths in metres, angles in radians. '
                       'Unknown mechanical limits and complete-body inertias remain null. '
                       'Motor CAD meshes are nominal visual geometry only. Do not enable hardware from this draft.',
                  sources=['https://github.com/wuphilipp/gello_mechanical',
                           'https://www.robotis.com/service/download.php?no=1986',
                           'https://www.robotis.com/service/download.php?no=1987'])
    # Base STL lies in XZ; turn it onto XY and put its first shaft projection at origin.
    raw_from_link = [np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])]
    raw_origins = [np.array([-70., 0, 0])]
    for point, normal, guide in HORNS:
        raw_origins.append(np.asarray(point, float))
        raw_from_link.append(frame(normal, guide))
    for i, link in enumerate(config['links']):
        rotation, point = raw_from_link[i], raw_origins[i]
        link['visuals'][0].update(xyz=(-rotation.T @ point * .001).tolist(), rpy=rpy(rotation.T))
        link['datum_status'] = 'STL-derived mating plane; zero clocking provisional'
        if i:
            link['visuals'].append({'mesh': 'motors/XL330_horn.STL',
                                   'xyz': [0., 0., -.0065], 'rpy': [0., 0., 0.],
                                   'role': 'servo_horn', 'rgba': '0.55 0.57 0.60 1',
                                   'note': 'Output horn follows this child link; CAD horn mating face z=6.5 mm'})
    for i, (case, long, normal, guide, width) in enumerate(CASES):
        case, long = np.asarray(case, float), np.asarray(long, float)
        long /= np.linalg.norm(long)
        normal = np.asarray(normal, float)
        normal /= np.linalg.norm(normal)
        axis_point = case + 7.5*long + 26*normal
        outgoing = frame(normal, guide)
        angle = np.deg2rad(clock[i])
        outgoing = outgoing @ np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        parent_rotation = raw_from_link[i]
        config['joints'][i].update(
            xyz=(parent_rotation.T @ (axis_point-raw_origins[i])*.001).tolist(),
            rpy=rpy(parent_rotation.T @ outgoing), axis=[0., 0., 1.],
            datum_status='axis from bolt pattern + nominal motor depth; shaft-end choice provisional',
            motor_rear_case_raw_mm=case.tolist(), shaft_axis_point_raw_mm=axis_point.tolist())
        # Vendor CAD rear case datum is [0,-7.5,-19.5] mm; horn face z=6.5.
        # Case Y points towards its output shaft, with a right-handed XYZ frame.
        case_rotation = np.column_stack((np.cross(long, normal), long, normal))
        case_origin = case-case_rotation @ np.array([0., -7.5, -19.5])
        config['links'][i]['visuals'].append({
            'mesh': 'motors/XL330_body.STL', 'role': 'servo_case', 'rgba': '0.22 0.24 0.28 1',
            'xyz': (parent_rotation.T @ (case_origin-raw_origins[i])*.001).tolist(),
            'rpy': rpy(parent_rotation.T @ case_rotation),
            'note': 'Nominal ROBOTIS CAD case geometry; not a mass/inertia prior'})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-geometry', required=True)
    parser.add_argument('--output-urdf', required=True)
    parser.add_argument('--clock-degrees', type=float, nargs=7, default=REFERENCE_CLOCK_DEGREES,
                        help='Virtual horn clocking; adjusting these does not calibrate real encoders')
    args = parser.parse_args()
    if any(Path(p).exists() for p in (args.output_geometry, args.output_urdf)):
        parser.error('Use new output paths; existing geometry/model will not be overwritten')
    config = reconstruct(args.clock_degrees)
    robot = build_model(config, draft=True)
    Path(args.output_geometry).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_geometry).open('x') as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    write_xml(robot, args.output_urdf)
    print(f'Saved offline draft: {args.output_geometry}, {args.output_urdf}')


if __name__ == '__main__':
    main()
