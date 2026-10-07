#!/usr/bin/env python3
"""Build a seven-axis visual/dynamics URDF with the official G2 frozen in place."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'gello_teleop/models'
XACRO = '{http://ros.org/wiki/xacro}'
COMMIT = '62936f7ea1846a85f7350de2c4c18f39e6d19715'


def geometry(link, mesh, *, grey=False):
    for tag in ('visual', 'collision'):
        element = ET.SubElement(link, tag)
        ET.SubElement(element, 'origin', xyz='0 0 0', rpy='0 0 0')
        g = ET.SubElement(element, 'geometry')
        ET.SubElement(g, 'mesh', filename=mesh)
        if tag == 'visual':
            material = ET.SubElement(element, 'material', name='g2_metal' if grey else 'xarm_white')
            ET.SubElement(material, 'color', rgba='.65 .65 .65 1' if grey else '.85 .85 .85 1')


def build(output, *, gripper_angle=.85):
    if not np.isfinite(gripper_angle) or not 0 <= gripper_angle <= .85:
        raise ValueError('gripper_angle must be within official drive_joint range [0,0.85] rad')
    root = ET.parse(MODELS/'xarm7_dynamics.urdf').getroot()
    if [j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed'] != [f'joint{i}' for i in range(1,8)]:
        raise ValueError('Base URDF must contain exactly joint1..joint7')
    root.set('name', 'xarm7_with_g2_fixed')
    root.insert(0, ET.Comment(
        f'UFACTORY xarm_ros2 {COMMIT}, BSD-3-Clause (LICENSE.xarm_ros2). '
        f'G2 visual meshes; upstream shared G1/G2 inertial defaults (not calibrated G2 inertias). '
        f'All six gripper mechanism joints frozen at drive_joint={gripper_angle:.8g} rad. '
        'Only joint1..joint7 are movable. link_eef retains the training flange frame; '
        'link_tcp is the upstream nominal gripper TCP.'))
    for link in root.findall('link'):
        name=link.get('name')
        if name == 'link_base' or name in {f'link{i}' for i in range(1,8)}:
            geometry(link, f'../meshes/xarm7_g2/xarm7/visual/{name}.stl', grey=name=='link7')
    upstream = ET.parse(MODELS/'sources/xarm_gripper.urdf.xacro').getroot()
    macro = upstream.find(XACRO+'macro')
    fix = ET.SubElement(root, 'joint', name='gripper_fix', type='fixed')
    ET.SubElement(fix, 'parent', link='link_eef')
    ET.SubElement(fix, 'child', link='xarm_gripper_base_link')
    ET.SubElement(fix, 'origin', xyz='0 0 0', rpy='0 0 0')
    for original in macro:
        if original.tag not in ('link', 'joint'):
            continue
        node=copy.deepcopy(original)
        for child in node.iter():
            for key,value in list(child.attrib.items()):
                child.set(key,value.replace('${prefix}','').replace('${name_suffix}','_g2'))
        if node.tag == 'link':
            for child in list(node):
                if child.tag.startswith(XACRO):
                    node.remove(child)
            visual = original.find(XACRO+'common_link_visual')
            if visual is not None:
                mesh = visual.get('mesh_filename').replace('${name_suffix}','_g2')
                geometry(node, f'../meshes/xarm7_g2/{mesh}.dae', grey=node.get('name')!='xarm_gripper_base_link')
        elif node.get('type') != 'fixed':
            axis=np.fromstring(node.find('axis').get('xyz'),sep=' ')
            origin=node.find('origin')
            initial=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
            rotated=initial * Rotation.from_rotvec(axis*gripper_angle)
            origin.set('rpy',' '.join(f'{v:.16g}' for v in rotated.as_euler('xyz')))
            node.set('type','fixed')
            for child in list(node):
                if child.tag in ('axis','limit','mimic','dynamics'):
                    node.remove(child)
        root.append(node)
    for node in root.iter():
        if any('${' in value for value in node.attrib.values()):
            raise ValueError('Unresolved xacro expression')
    ET.indent(root,space='  ')
    output=Path(output)
    output.write_text("<?xml version='1.0' encoding='utf-8'?>\n"+ET.tostring(root,encoding='unicode')+'\n')
    print(f'Built {output}; G2 fixed at {gripper_angle} rad; seven movable joints')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=MODELS/'xarm7_dynamics_g2.urdf')
    parser.add_argument('--gripper-angle',type=float,default=.85)
    args=parser.parse_args()
    build(args.output,gripper_angle=args.gripper_angle)


if __name__=='__main__':
    main()
