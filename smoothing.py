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
smpl_hand_faces = np.load('/home/chen/IPNet_SMPLX/assets/smpl_hand_faces.npy')
if __name__ == "__main__":

    device = torch.device('cuda')

    max_collisions = 8
        
    batch_size = 1
    
    out_dir = './out_dir15/'
    gender = 'male'
    model_path = out_dir + 'full_smpld.pkl'
    model = pkl.load(open(model_path, 'rb'), encoding='latin1')
    
    num_iter = 0
    while True:
        
        if num_iter == 0:
            mesh_fn = out_dir + 'full_smpld.obj'
            ps_mesh = Mesh(filename=mesh_fn)
            input_mesh = trimesh.Trimesh(vertices=ps_mesh.v, faces=ps_mesh.f, process=False)
            v_ori = ps_mesh.v.copy()
        else:    
            mesh_fn = out_dir + 'full_smpld_sm.obj'
            ps_mesh = Mesh(filename=mesh_fn)
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

        excl_collisions = []
        # hand faces excluded
        for c_fs in collisions:
            if c_fs[0] not in smpl_hand_faces and c_fs[1] not in smpl_hand_faces:
                excl_collisions.append(c_fs)
        excl_collisions = np.array(excl_collisions)

        print(excl_collisions.shape)

        print('Number of collisions = ', excl_collisions.shape[0])
        if excl_collisions.shape[0] == 0:
            break
        print('Percentage of collisions (%)',
            excl_collisions.shape[0] / float(triangles.shape[1]) * 100)        
        # recv_faces = input_mesh.faces[collisions[:, 0]]
        # intr_faces = input_mesh.faces[collisions[:, 1]]
        recv_faces = input_mesh.faces[excl_collisions[:, 0]]
        intr_faces = input_mesh.faces[excl_collisions[:, 1]]

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
        trimesh.exchange.export.export_mesh(input_mesh, out_dir + 'full_smpld_sm.obj')

        num_iter += 1
    # pdb.set_trace()
    mesh_fn = out_dir + 'full_smpld_sm.obj'
    final_mesh = trimesh.load(mesh_fn, process=False)

    

    smpl = th_SMPL(betas = torch.tensor(model['betas']), trans=torch.tensor(model['trans']), pose = torch.tensor(model['pose']), gender=gender)
    vertices, th_T = smpl.forward(return_lbs = True)
    # smpl = th_SMPL(betas = torch.tensor(model['betas']), gender=gender)
    # vertices = smpl.forward(return_lbs = False)

    # pdb.set_trace()
    temp_verts = torch.cat([vertices.type(th_T.dtype), torch.ones((final_mesh.vertices.shape[0], 1), dtype=th_T.dtype, device=th_T.device),], 1).unsqueeze(-1)
    temp_verts_T = torch.matmul(th_T.transpose(0,3)[..., 0].inverse(), temp_verts)
    
    homo_verts = torch.cat([torch.tensor(final_mesh.vertices).type(th_T.dtype), torch.ones((final_mesh.vertices.shape[0], 1), dtype=th_T.dtype, device=th_T.device),], 1).unsqueeze(-1)
    homo_verts_T = torch.matmul(th_T.transpose(0,3)[..., 0].inverse(), homo_verts)
    offsets_sm_T = homo_verts_T[:, :3, 0].detach().cpu().numpy() - temp_verts_T[:, :3, 0].detach().cpu().numpy()



    # offsets_sm = final_mesh.vertices - vertices.detach().cpu().numpy() # v_ori + model['offsets'].copy()

    # homo_offsets_sm = torch.cat([torch.tensor(offsets_sm).type(th_T.dtype), torch.ones((final_mesh.vertices.shape[0], 1), dtype=th_T.dtype, device=th_T.device),], 1).unsqueeze(-1)
    # homo_offsets_sm_T = torch.matmul(th_T.transpose(0,3)[..., 0].inverse(), homo_offsets_sm)
    # offsets_sm_T = homo_offsets_sm_T[:, :3, 0].detach().cpu().numpy()

    offsets_sm_T *= binary_mask
    # smpl_sm_dict = {'pose': model['pose'], 'betas': model['betas'], 'trans': model['trans'], 'offsets': offsets_sm}
    smpl_sm_dict = {'pose': model['pose'], 'betas': model['betas'], 'trans': model['trans'], 'offsets': offsets_sm_T}

    pkl.dump(smpl_sm_dict, open(out_dir+'full_smpld_sm.pkl', 'wb'))
    test_model_path = out_dir + 'full_smpld_sm.pkl'
    test_model = pkl.load(open(test_model_path, 'rb'), encoding='latin1')

    # test_smpl = th_SMPL(betas = torch.tensor(test_model['betas']), trans = torch.tensor(test_model['trans']), offsets = torch.tensor(model['offsets']), gender=gender)
    # test_vertices = test_smpl.forward()
    # test_mesh = trimesh.Trimesh(test_vertices.detach().cpu().numpy(), input_mesh.faces, process=False)

    # trimesh.exchange.export.export_mesh(test_mesh, out_dir + 'test.obj')

    test_smpl = th_SMPL(betas = torch.tensor(test_model['betas']), trans = torch.tensor(test_model['trans']), offsets = torch.tensor(test_model['offsets']), gender=gender)
    test_vertices = test_smpl.forward()
    test_mesh = trimesh.Trimesh(test_vertices.detach().cpu().numpy(), input_mesh.faces, process=False)

    trimesh.exchange.export.export_mesh(test_mesh, out_dir + 'full_smpld_sm_T.obj')