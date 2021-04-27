"""
This code optimizes the offsets on top of SMPL.
If code works:
    Author: Bharat
else:
    Author: Anonymous
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""
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

def interpenetration_loss(vertices=None, faces=None, search_tree=None, pen_distance=None, tri_filtering_module=None, coll_loss_weight=0.0):
    pen_loss = torch.tensor(0.0).unsqueeze(0).cuda()
    batch_size = 1 # projected_joints.shape[0] 

    triangles = torch.index_select(vertices, 1, faces.type(torch.LongTensor).cuda().view(-1)).view(1, -1, 3, 3)

    with torch.no_grad():
        collision_idxs = search_tree(triangles)

    # Remove unwanted collisions
    if tri_filtering_module is not None:
        collision_idxs = tri_filtering_module(collision_idxs)

    if collision_idxs.ge(0).sum().item() > 0:
        pen_loss = torch.sum(
            coll_loss_weight *
            pen_distance(triangles, collision_idxs)).unsqueeze(0)

    return pen_loss

def get_loss_weights():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst, #/ (1 + it),
                   'lap': lambda cst, it: 5. ** 4 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** 1 * cst / (1 + it)
                   # 'normal': lambda cst, it: 10 ** -5 * cst/ (1 + it)
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
    loss['offsets'] = torch.mean(torch.mean(smplx.offsets**2, axis=1), axis=1)
    # loss['normal'] = mesh_normal_consistency(p3d_meshes).unsqueeze(0)
    # loss['interpenetration'] = interpenetration_loss(verts, smplx.faces, search_tree, pen_distance, tri_filtering_module, 1.0)
    return loss


def optimize_offsets(th_scan_meshes, smplx, init_smplx_meshes, iterations, steps_per_iter, search_tree, pen_distance, tri_filtering_module):
    # Optimizer
    optimizer = torch.optim.Adam([smplx.offsets, smplx.trans, smplx.betas, smplx.global_pose, smplx.body_pose,
                                  smplx.left_hand_pose, smplx.right_hand_pose], 0.005, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL+D')
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


def fit_SMPLXD(scans, smpl_pkl=None, gender='male', save_path=None, display=False):
    # Get SMPL faces
    sp = SmplPaths(gender=gender)
    smpl_faces = sp.get_faces()
    th_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()

    # Batch size
    batch_sz = len(scans)

    # Init SMPL
    if smpl_pkl is None or smpl_pkl[0] is None:
        print('SMPL not specified, fitting SMPL now')
        pose, betas, trans = fit_SMPLX(scans, None, gender, save_path, display)
    else:
        pose, betas, trans = [], [], []
        for spkl in smpl_pkl:
            smpl_dict = pkl.load(open(spkl, 'rb'), encoding='latin-1')
            p, b, t = smpl_dict['pose'], smpl_dict['betas'], smpl_dict['trans']
            pose.append(p)
            if len(b) == 10:
                temp = np.zeros((300,))
                temp[:10] = b
                b = temp.astype('float32')
            betas.append(b)
            trans.append(t)
        pose, betas, trans = np.array(pose), np.array(betas), np.array(trans)

    betas, pose, trans = torch.tensor(betas), torch.tensor(pose), torch.tensor(trans)
    smplx = th_batch_SMPLX(batch_sz, betas, pose, trans, faces=th_faces).cuda()

    verts, _, _, _ = smplx()
    init_smplx_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                        faces=smplx.faces) for v in verts]

    # Load scans
    th_scan_meshes = []
    for scan in scans:
        th_scan = tm.from_obj(scan)
        if save_path is not None:
            th_scan.save_mesh(join(save_path, split(scan)[1]))
        th_scan.vertices = th_scan.vertices.cuda()
        th_scan.faces = th_scan.faces.cuda()
        th_scan.vertices.requires_grad = False
        th_scan_meshes.append(th_scan)

    # Optimize
    optimize_offsets(th_scan_meshes, smplx, init_smplx_meshes, 5, 10)
    print('Done')

    verts, _, _, _ = smplx()
    th_SMPLX_meshes = [tm.from_tensors(vertices=v,
                                      faces=smplx.faces) for v in verts]

    if save_path is not None:
        if not exists(save_path):
            os.makedirs(save_path)

        names = [split(s)[1] for s in scans]

        # Save meshes
        save_meshes(th_SMPLX_meshes, [join(save_path, n.replace('.obj', '_smpld.obj')) for n in names])
        save_meshes(th_scan_meshes, [join(save_path, n) for n in names])
        # Save params
        for p, b, t, d, n in zip(smplx.pose.cpu().detach().numpy(), smplx.betas.cpu().detach().numpy(),
                                 smplx.trans.cpu().detach().numpy(), smplx.offsets.cpu().detach().numpy(), names):
            smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'offsets': d}
            pkl.dump(smpl_dict, open(join(save_path, n.replace('.obj', '_smpld.pkl')), 'wb'))

    return smplx.pose.cpu().detach().numpy(), smplx.betas.cpu().detach().numpy(), \
           smplx.trans.cpu().detach().numpy(), smplx.offsets.cpu().detach().numpy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('-smpl_pkl', type=str, default=None)  # In case SMPL fit is already available
    parser.add_argument('-gender', type=str, default='male')  # can be female/ male/ neutral
    parser.add_argument('--display', default=False, action='store_true')
    args = parser.parse_args()

    # args = lambda: None
    # args.scan_path = '/BS/bharat-2/static00/renderings/renderpeople/rp_alison_posed_017_30k/rp_alison_posed_017_30k.obj'
    # args.smpl_pkl = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/rp_alison_posed_017_30k_smpl.pkl'
    # args.display = False
    # args.save_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data'
    # args.gender = 'female'

    _, _, _, _ = fit_SMPLXD([args.scan_path], smpl_pkl=[args.smpl_pkl], display=args.display, save_path=args.save_path,
                           gender=args.gender)
