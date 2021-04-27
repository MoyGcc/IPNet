"""
Scans need to be processed before passing them to IPNet.
Users would have to modify the script a bit to suit their I/O. function func can be directly used.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""

import os
from os.path import join, split, exists
from shutil import copyfile
from glob import glob
from psbody.mesh import Mesh
import trimesh
import numpy as np

bb_min = -1.
bb_max = 1.

new_cent = (bb_max + bb_min) / 2
SCALE = 1.5
MIN_SCALE = 1.6


def func(vertices, scale=None, cent=None):
    """
    Function to normalize the scans for IPNet.
    Ensure that the registration and body are normalized in the same way as scan.
    """
    # import pdb
    # pdb.set_trace()
    if scale is None:
        scale = max(MIN_SCALE, vertices[:, 1].max() - vertices[:, 1].min())
    
    vertices /= (scale / SCALE)

    if cent is None:
        cent = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices -= (cent - new_cent)

    return vertices, scale, cent


def process(scan, name, path):
    # Load scan and normalize
    # mesh = trimesh.load(join(scan, name + '.obj'), process=False)
    mesh = Mesh(filename=join(scan, name + '.obj'))
    mesh.v, scale, cent = func(mesh.v)
    # mesh.vt = mesh.vt[:, :2]
    mesh.write_obj(join(path, name + '_scaled.obj'))
    np.save(join(path, name + '_cent.npy'), [scale / SCALE, (cent - new_cent)])

    # Normalize the registration according to scan.
    if exists(join(scan, name + '_reg.obj')):
        reg = Mesh(filename=join(scan, name + '_reg.obj'))  # Not using trimesh as it changes vertex ordering
        reg.v, _, _ = func(reg.v, scale, cent)
        reg.write_obj(join(path, name + '_scaled_reg.obj'))
        count_reg = 1
    else:
        count_reg = 0

    # normalize the body under clothing.

    if exists(join(scan, name + '_naked.obj')):
        mesh = Mesh(filename=join(scan, name + '_naked.obj'))  # Not using trimesh as it changes vertex ordering
        mesh.v, _, _ = func(mesh.v, scale, cent)
        mesh.write_obj(join(path, name + '_scaled_naked.obj'))
        count_body = 1
    else:
        count_body = 0

    print('Done,', name)
    return count_reg, count_body


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('scan', type=str, default='out_dir')
    # parser.add_argument('body', type=str)
    parser.add_argument('name', type=str, default='pifu_mono_4')
    parser.add_argument('path', type=str, default='out_dir/')
    args = parser.parse_args()

    _, _ = process(args.scan, args.name, args.path)
