from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time
import pickle as pkl
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
import pdb
from lib.th_SMPL import th_SMPL

smpl_right_hand_idx = np.load('/home/chen/IPNet_SMPLX/assets/smpl_right_hand_idx.npy')
smpl_left_hand_idx = np.load('/home/chen/IPNet_SMPLX/assets/smpl_left_hand_idx.npy')
binary_mask = np.ones((6890, 3))
# binary_mask[:,smpl_face_idx,:] = 0.
binary_mask[smpl_right_hand_idx,:] = 0.
binary_mask[smpl_left_hand_idx,:] = 0.

if __name__ == "__main__":

    device = torch.device('cuda')

    max_collisions = 8
        
    batch_size = 1
    
    out_dir = './out_dir18/'
    gender = 'female'
    model_path = out_dir + 'full_smpld.pkl'
    model = pkl.load(open(model_path, 'rb'), encoding='latin1')

    smpl = th_SMPL(betas = torch.tensor(model['betas']), trans = torch.tensor(model['trans']), offsets = torch.tensor(model['offsets']), gender=gender)
    vertices = smpl.forward()

    mesh_fn = out_dir + 'full_smpld.obj'
    ps_mesh = Mesh(filename=mesh_fn)

    T_mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), ps_mesh.f, process=False)
    trimesh.exchange.export.export_mesh(T_mesh, out_dir + 'full_smpld_T.obj')
    num_iter = 0

    while True:
        
        if num_iter == 0:
            mesh_fn = out_dir + 'full_smpld_T.obj'
            ps_mesh = Mesh(filename=mesh_fn)
            input_mesh = trimesh.Trimesh(vertices=ps_mesh.v, faces=ps_mesh.f, process=False)
            # input_mesh = trimesh.load(mesh_fn, process=False, maintain_order=True)
            v_ori = input_mesh.vertices.copy()
        else:    
            mesh_fn = out_dir + 'full_smpld_T_sm.obj'
            ps_mesh = Mesh(filename=mesh_fn)
            # input_mesh = trimesh.Trimesh(vertices=ps_mesh.v, faces=ps_mesh.f, maintain_order=True)             
            input_mesh = trimesh.load(mesh_fn, process=False)
        
        vertices = torch.tensor(input_mesh.vertices,
                                dtype=torch.float32, device=device)
        faces = torch.tensor(input_mesh.faces.astype(np.int64),
                            dtype=torch.long,
                            device=device)

        
        triangles = vertices[faces].unsqueeze(dim=0)

        torch.cuda.synchronize()
        start = time.time()
        m = BVH(max_collisions=max_collisions)
        outputs = m(triangles)
        torch.cuda.synchronize()
        print('Elapsed time', time.time() - start)

        outputs = outputs.detach().cpu().numpy().squeeze()

        collisions = outputs[outputs[:, 0] >= 0, :]

        print(collisions.shape)

        print('Number of collisions = ', collisions.shape[0])
        if collisions.shape[0] == 0:
            break
        print('Percentage of collisions (%)',
            collisions.shape[0] / float(triangles.shape[1]) * 100)
        recv_faces = input_mesh.faces[collisions[:, 0]]
        intr_faces = input_mesh.faces[collisions[:, 1]]
        collisions_faces = np.concatenate((recv_faces, intr_faces))
        uniques, _ = np.unique(collisions_faces, axis=0, return_index=True)
        sorted_uniques = np.unique(uniques)
        values = np.arange(len(sorted_uniques))  
        norm_uniques = np.zeros_like(uniques)

        table = {}

        for u in range(len(sorted_uniques)):
            table[sorted_uniques[u]] = values[u]
        for i in range(uniques.shape[0]):
            for j in range(uniques.shape[1]):
                 norm_uniques[i,j] = table[uniques[i,j]]
        # uniques = collisions_faces[np.sort(unique_index)]
        # pdb.set_trace()
        # r_mesh = trimesh.Trimesh(input_mesh.vertices, recv_faces, process=False)
        # i_mesh = trimesh.Trimesh(input_mesh.vertices, intr_faces, process=False)

        referenced = np.zeros(len(input_mesh.vertices), dtype=bool)
        # referenced[collisions_faces] = True
        referenced[uniques] = True
        mask = np.where(referenced==True)[0]
        # pdb.set_trace()
        collisions_mesh = trimesh.Trimesh(input_mesh.vertices[sorted_uniques].copy(), norm_uniques, process=False)

        trimesh.smoothing.filter_humphrey(collisions_mesh, alpha=0.1, beta=0.5, iterations=20)
        input_mesh.vertices[mask] = collisions_mesh.vertices
        # ps_mesh = Mesh(v = input_mesh.vertices, f = ps_mesh.f)
        # pdb.set_trace()
        # ps_mesh.write_obj(out_dir + 'full_smpld_sm.obj')
        # pdb.set_trace()
        trimesh.exchange.export.export_mesh(input_mesh, out_dir + 'full_smpld_T_sm.obj')

        num_iter += 1
    # pdb.set_trace()
    mesh_fn = out_dir + 'full_smpld_T_sm.obj'
    final_mesh = trimesh.load(mesh_fn, process=False)

    


    # pdb.set_trace()
    smpl = th_SMPL(betas = torch.tensor(model['betas']), trans = torch.tensor(model['trans']), gender=gender)
    vertices = smpl.forward()
    offsets_sm = final_mesh.vertices - vertices.detach().cpu().numpy() # v_ori + model['offsets'].copy()
    # offsets_sm *= binary_mask
    smpl_sm_dict = {'pose': model['pose'], 'betas': model['betas'], 'trans': model['trans'], 'offsets': offsets_sm}

    pkl.dump(smpl_sm_dict, open(out_dir+'full_smpld_sm.pkl', 'wb'))

    model_path = out_dir + 'full_smpld_sm.pkl'
    model = pkl.load(open(model_path, 'rb'), encoding='latin1')

    smpl = th_SMPL(betas = torch.tensor(model['betas']), pose = torch.tensor(model['pose']), trans = torch.tensor(model['trans']), offsets = torch.tensor(model['offsets']), gender=gender)
    vertices = smpl.forward()
    test_mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), ps_mesh.f, process=False)
    trimesh.exchange.export.export_mesh(test_mesh, out_dir + 'test.obj')