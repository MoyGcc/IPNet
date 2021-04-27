from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import argparse

try:
    input = raw_input
except NameError:
    pass

import torch
import torch.nn as nn
import torch.autograd as autograd

from copy import deepcopy

import numpy as np
import tqdm

import trimesh
from psbody.mesh import Mesh
import pyrender
from mesh_intersection.bvh_search_tree import BVH

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes

if __name__ == "__main__":

    device = torch.device('cuda')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('mesh_fn', type=str,
    #                     help='A mesh file (.obj, .ply, e.t.c.) to be checked' +
    #                     ' for collisions')
    # parser.add_argument('--max_collisions', default=8, type=int,
    #                     help='The maximum number of bounding box collisions')

    # args, _ = parser.parse_known_args()

    # mesh_fn = args.mesh_fn
    # max_collisions = args.max_collisions

    mesh_fn = './out_dir28/027_1_2.obj'
    max_collisions = 8
    input_mesh = trimesh.load(mesh_fn)
    # trimesh.smoothing.filter_humphrey(input_mesh, iterations=2)
    # trimesh.repair.fix_winding(input_mesh)
    
    # import pdb
    # pdb.set_trace()

    input_mesh = trimesh.graph.split(input_mesh)[0]
    trimesh.exchange.export.export_mesh(input_mesh, './out_dir28/027_1_2'+'_cleaned.obj')


    # pifu_fn = './out_dir6/helge_scaled.obj'
    # pifu_mesh = Mesh(filename=pifu_fn)

    # p3d_mesh = Meshes(verts=[torch.tensor(pifu_mesh.v, dtype=torch.float32, device=device)], faces=[torch.tensor(pifu_mesh.f.astype(np.int64), dtype=torch.long, device=device)])
    # subdivide = SubdivideMeshes()
    # # samples = sample_points_from_meshes(p3d_mesh, num_samples=60000)
    # import pdb
    # pdb.set_trace()
    # sub_p3d_mesh = subdivide(p3d_mesh)
    # final_verts, final_faces = sub_p3d_mesh.get_mesh_verts_faces(0)
    # save_obj('./subdivided_helge_scaled.obj', final_verts, final_faces)
    print('Number of triangles = ', input_mesh.faces.shape[0])

    vertices = torch.tensor(input_mesh.vertices,
                            dtype=torch.float32, device=device)
    faces = torch.tensor(input_mesh.faces.astype(np.int64),
                         dtype=torch.long,
                         device=device)

    batch_size = 1
    triangles = vertices[faces].unsqueeze(dim=0)

    m = BVH(max_collisions=max_collisions)

    torch.cuda.synchronize()
    start = time.time()
    outputs = m(triangles)
    torch.cuda.synchronize()
    print('Elapsed time', time.time() - start)

    outputs = outputs.detach().cpu().numpy().squeeze()

    collisions = outputs[outputs[:, 0] >= 0, :]

    print(collisions.shape)

    print('Number of collisions = ', collisions.shape[0])
    print('Percentage of collisions (%)',
          collisions.shape[0] / float(triangles.shape[1]) * 100)
    recv_faces = input_mesh.faces[collisions[:, 0]]
    intr_faces = input_mesh.faces[collisions[:, 1]]
    # collisions_faces = np.concatenate((recv_faces, intr_faces))
    import pdb

    r_mesh = trimesh.Trimesh(input_mesh.vertices, recv_faces)
    i_mesh = trimesh.Trimesh(input_mesh.vertices, intr_faces)
    


    # from trimesh.util import concatenate, decimal_to_digits
    # from trimesh.constants import tol
    # from trimesh.grouping import unique_rows
    
    # digits_vertex = decimal_to_digits(tol.merge)
    # referenced = np.zeros(len(input_mesh.vertices), dtype=bool)
    # referenced[collisions_faces] = True
    # mask = np.where(referenced==True)[0]

    # stacked = [input_mesh.vertices * (10 ** digits_vertex)]
    # stacked = np.column_stack(stacked).round().astype(np.int64)
    # u, i = unique_rows(stacked[referenced])
    # inverse = np.zeros(len(input_mesh.vertices), dtype=np.int64)
    # inverse[referenced] = i
    # mask = np.nonzero(referenced)[0][u]
    # collisions_mesh = trimesh.Trimesh(input_mesh.vertices, collisions_faces)

    # trimesh.smoothing.filter_humphrey(collisions_mesh, iterations=20)

    # input_mesh.vertices[mask] = collisions_mesh.vertices
    

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 0.99])
    recv_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.0, 0.9, 0.0, 1.0])
    intr_material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=[0.9, 0.0, 0.0, 1.0])

    main_mesh = pyrender.Mesh.from_trimesh(input_mesh, material=material)

    recv_mesh = pyrender.Mesh.from_trimesh(
        r_mesh,
        material=recv_material)
    intr_mesh = pyrender.Mesh.from_trimesh(
        i_mesh,
        material=intr_material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(main_mesh)
    scene.add(recv_mesh)
    scene.add(intr_mesh)

    pyrender.Viewer(scene, use_raymond_lighting=True, cull_faces=False)