import os
from os.path import split, join, exists
from glob import glob
import torch
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import point_to_surface, laplacian_loss

from tqdm import tqdm
import pickle as pkl
import numpy as np

# from lib.smpl_paths import SmplPaths
from lib.th_SMPLX import th_batch_SMPLX

from fit_SMPLX import fit_SMPLX, save_meshes, batch_point_to_surface, backward_step

from pytorch3d.structures import Meshes 
from pytorch3d.loss.mesh_normal_consistency import mesh_normal_consistency


def get_loss_weights():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst, #/ (1 + it),
                   'lap': lambda cst, it: 10. ** 4 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** 1 * cst / (1 + it),
                   'normal': lambda cst, it: 10 ** -5 * cst/ (1 + it)
                   }
                   
    return loss_weight

def forward_step(th_scan_meshes, smplx, init_smplx_meshes, search_tree, pen_distance, tri_filtering_module):
    """
    Performs a forward step, given smplx and scan meshes.
    Then computes the losses.
    """

    # forward
    # verts, _, _, _ = smplx()
    verts = smplx()
    th_SMPLX_meshes = [tm.from_tensors(vertices=v,
                                      faces=smplx.faces) for v in verts]
    p3d_meshes = Meshes(verts=verts, faces=smplx.faces.expand(1,-1,-1))
    # losses
    loss = dict()
    loss['s2m'] = batch_point_to_surface([sm.vertices for sm in th_scan_meshes], th_SMPLX_meshes)
    loss['m2s'] = batch_point_to_surface([sm.vertices for sm in th_SMPLX_meshes], th_scan_meshes)
    loss['lap'] = torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(init_smplx_meshes, th_SMPLX_meshes)])
    # loss['offsets'] = torch.mean(torch.mean(smplx.offsets**2, axis=1), axis=1)
    # loss['normal'] = mesh_normal_consistency(p3d_meshes).unsqueeze(0)
    # loss['interpenetration'] = interpenetration_loss(verts, smplx.faces, search_tree, pen_distance, tri_filtering_module, 1.0)
    return loss

def optimize_offsets_only(th_scan_meshes, smplx, init_smplx_meshes, iterations, steps_per_iter, search_tree, pen_distance, tri_filtering_module):
    # Optimizer
    optimizer = torch.optim.Adam([smplx.offsets], 0.005, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing only D')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, smplx, init_smplx_meshes, search_tree, pen_distance, tri_filtering_module)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Lx100. Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item()*100)
            loop.set_description(l_str)
