#!/usr/bin/env python3
"""Build a measured GELLO model, or an explicitly unassembled STL preview."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = ROOT / 'mesh/gello_mechanical'


def vector(value, name, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: supply {size} measured, finite numbers')
    return result


def numbers(value):
    return ' '.join(f'{x:.12g}' for x in value)


def stl_bounds(path):
    content = Path(path).read_bytes()
    if len(content) < 84:
        raise ValueError(f'Not a binary STL: {path}')
    count = struct.unpack_from('<I', content, 80)[0]
    if not count or len(content) != 84 + count * 50:
        raise ValueError(f'Invalid binary STL: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                      ('attribute', '<u2')])
    vertices = np.frombuffer(content, dtype=dtype, offset=84)['vertices'].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f'Nonfinite STL vertices: {path}')
    return vertices.min(axis=0), vertices.max(axis=0)


def template():
    """Names/mesh assignments are a proposed partition, to confirm in the CAD."""
    names = ['base_link'] + [f'link{i}' for i in range(1, 8)]
    files = ['xarm7/base.STL'] + [f'xarm7/L{i}.STL' for i in range(1, 7)] + ['gripper/handle.STL']
    links = []
    for name, file in zip(names, files):
        links.append({'name': name, 'visuals': [{'mesh': file, 'xyz': None, 'rpy': None}],
                      'inertial': None})
    joints = [{'name': f'joint{i}', 'parent': names[i - 1], 'child': names[i],
               'xyz': None, 'rpy': None, 'axis': None,
               'lower': None, 'upper': None, 'effort': None, 'velocity': None}
              for i in range(1, 8)]
    return {'robot_name': 'gello_xarm7', 'geometry_verified': False,
            'mesh_units_verified': False, 'mesh_scale': 0.001,
            'note': 'Confirm link partition, servo case/horn attachments and zero pose. '
                    'xyz is metres; rpy/limits are radians. Add motor/support visuals as needed. '
                    'Handle/locked trigger and other loads must be included in link7 inertia.',
            'links': links, 'joints': joints}


def add_visual(link, path, xyz, rpy, scale, color=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    visual = ET.SubElement(link, 'visual')
    ET.SubElement(visual, 'origin', xyz=numbers(xyz), rpy=numbers(rpy))
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'mesh', filename=path.as_uri(), scale=numbers([scale] * 3))
    if color:
        material = ET.SubElement(visual, 'material', name=link.get('name')+'_'+path.stem+'_material')
        ET.SubElement(material, 'color', rgba=color)


def add_inertial(link, values):
    mass = float(values['mass_kg'])
    com = vector(values['com_m'], f'{link.attrib["name"]} COM')
    inertia = np.asarray(values['inertia_kg_m2'], dtype=float)
    if not np.isfinite(mass) or mass <= 0 or inertia.shape != (3, 3):
        raise ValueError('Inertia needs positive mass and a 3x3 COM inertia in link axes')
    if not np.isfinite(inertia).all() or not np.allclose(inertia, inertia.T, atol=1e-12):
        raise ValueError('Inertia must be finite and symmetric')
    eigen = np.linalg.eigvalsh(inertia)
    if eigen[0] <= 0 or eigen[-1] > eigen[0] + eigen[1] + 1e-12:
        raise ValueError('Inertia must be positive definite and satisfy the triangle inequality')
    element = ET.SubElement(link, 'inertial')
    ET.SubElement(element, 'origin', xyz=numbers(com), rpy='0 0 0')
    ET.SubElement(element, 'mass', value=f'{mass:.12g}')
    ET.SubElement(element, 'inertia', **{key: f'{inertia[i, j]:.12g}' for key, i, j in
                                       [('ixx', 0, 0), ('ixy', 0, 1), ('ixz', 0, 2),
                                        ('iyy', 1, 1), ('iyz', 1, 2), ('izz', 2, 2)]})


def build_model(config, require_inertia=False, draft=False):
    if not draft and (config.get('geometry_verified') is not True or config.get('mesh_units_verified') is not True):
        raise ValueError('Measure/confirm the geometry and STL units before building a moving model')
    if draft and require_inertia:
        raise ValueError('A geometry draft cannot certify dynamics')
    scale = float(config['mesh_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('mesh_scale must be positive')
    robot = ET.Element('robot', name=config['robot_name'] + ('_UNVERIFIED_DRAFT' if draft else ''))
    robot.append(ET.Comment('UNVERIFIED geometry draft. Continuous joints are for visualization only; '
                            'no measured motion ranges, encoder zero or dynamics.' if draft else
                            'Measured GELLO geometry; not the follower xArm model.'))
    links = config['links']
    expected = ['base_link'] + [f'link{i}' for i in range(1, 8)]
    if [link['name'] for link in links] != expected:
        raise ValueError(f'Expected link order {expected}')
    for values in links:
        link = ET.SubElement(robot, 'link', name=values['name'])
        for mesh in values['visuals']:
            if 'box_m' in mesh:
                visual = ET.SubElement(link, 'visual')
                ET.SubElement(visual, 'origin', xyz=numbers(vector(mesh['xyz'], 'box xyz')),
                              rpy=numbers(vector(mesh['rpy'], 'box rpy')))
                size = vector(mesh['box_m'], 'box dimensions')
                if np.any(size <= 0):
                    raise ValueError('Box dimensions must be positive')
                ET.SubElement(ET.SubElement(visual, 'geometry'), 'box', size=numbers(size))
                material = ET.SubElement(visual, 'material', name='servo_envelope')
                ET.SubElement(material, 'color', rgba='0.22 0.24 0.28 1')
            else:
                add_visual(link, MESH_ROOT / mesh['mesh'], vector(mesh['xyz'], 'mesh xyz'),
                           vector(mesh['rpy'], 'mesh rpy'), scale, color=mesh.get('rgba'))
        if values.get('inertial') is not None:
            add_inertial(link, values['inertial'])
        elif require_inertia:
            raise ValueError(f'{values["name"]}: supply the complete rigid body inertia')
    if len(config['joints']) != 7:
        raise ValueError('Expected seven arm joints; lock/include the trigger load in link7')
    for i, values in enumerate(config['joints'], 1):
        if (values['name'], values['parent'], values['child']) != (f'joint{i}', expected[i-1], expected[i]):
            raise ValueError(f'joint{i}: invalid parent/child chain')
        axis = vector(values['axis'], 'joint axis')
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            raise ValueError('Joint axis cannot be zero')
        if not draft:
            limits = vector([values[k] for k in ('lower', 'upper', 'effort', 'velocity')], 'limits', 4)
            if limits[0] >= limits[1] or np.any(limits[2:] <= 0):
                raise ValueError('Invalid revolute joint limits')
        joint = ET.SubElement(robot, 'joint', name=values['name'], type='continuous' if draft else 'revolute')
        ET.SubElement(joint, 'parent', link=values['parent'])
        ET.SubElement(joint, 'child', link=values['child'])
        ET.SubElement(joint, 'origin', xyz=numbers(vector(values['xyz'], 'joint xyz')),
                      rpy=numbers(vector(values['rpy'], 'joint rpy')))
        ET.SubElement(joint, 'axis', xyz=numbers(axis / norm))
        if not draft:
            ET.SubElement(joint, 'limit', **dict(zip(('lower', 'upper', 'effort', 'velocity'),
                                                  [f'{x:.12g}' for x in limits])))
    return robot


def build_preview():
    """A fixed, exploded mesh inventory, deliberately containing no movable joints."""
    robot = ET.Element('robot', name='gello_xarm7_UNASSEMBLED_PREVIEW')
    robot.append(ET.Comment('STL inventory only. Scale 0.001 is an unconfirmed mm assumption. '
                            'Transforms are display placements, NOT assembled geometry. No inertias.'))
    ET.SubElement(robot, 'link', name='preview_world')
    meshes = sorted((MESH_ROOT / 'xarm7').glob('*.STL')) + sorted((MESH_ROOT / 'gripper').glob('*.STL'))
    report = []
    for i, path in enumerate(meshes):
        minimum, maximum = stl_bounds(path)
        name = f'preview_{path.stem}'
        link = ET.SubElement(robot, 'link', name=name)
        add_visual(link, path, -(minimum + maximum) * 0.0005, [0, 0, 0], 0.001,
                   '0.65 0.75 0.85 1')
        joint = ET.SubElement(robot, 'joint', name=f'placement_{path.stem}', type='fixed')
        ET.SubElement(joint, 'parent', link='preview_world')
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz=numbers([(i % 3) * 0.25, (i // 3) * 0.18, 0]), rpy='0 0 0')
        report.append({'mesh': str(path.relative_to(ROOT)), 'min_raw': minimum.tolist(),
                       'max_raw': maximum.tolist(), 'size_raw': (maximum - minimum).tolist(),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return robot, report


def write_xml(robot, path):
    ET.indent(robot, space='  ')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        ET.ElementTree(robot).write(file, encoding='utf-8', xml_declaration=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    initialize = commands.add_parser('init', help='Write the geometry measurement template')
    initialize.add_argument('--output', required=True)
    preview = commands.add_parser('preview', help='Build the static STL inventory URDF')
    preview.add_argument('--output', required=True)
    preview.add_argument('--bounds-report')
    build = commands.add_parser('build', help='Build the seven-axis URDF from measured geometry')
    build.add_argument('--geometry', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--require-inertia', action='store_true')
    draft = commands.add_parser('draft', help='Offline moving geometry preview; no physical motion limits')
    draft.add_argument('--geometry', required=True)
    draft.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            with Path(args.output).open('x') as file:
                yaml.safe_dump(template(), file, sort_keys=False, allow_unicode=True)
        elif args.command == 'preview':
            robot, report = build_preview()
            write_xml(robot, args.output)
            if args.bounds_report:
                with Path(args.bounds_report).open('x') as file:
                    json.dump(report, file, indent=2)
        else:
            config = yaml.safe_load(Path(args.geometry).read_text())
            write_xml(build_model(config, getattr(args, 'require_inertia', False),
                                  draft=args.command == 'draft'), args.output)
        print(args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
