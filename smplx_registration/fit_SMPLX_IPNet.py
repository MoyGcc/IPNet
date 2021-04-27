"""
Code to fit SMPLX (pose, shape) to IPNet predictions using pytorch, kaolin.
Author: Bharat
Cite: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV 2020.
"""
import os
from os.path import split, join, exists
import sys
# import ipdb
import json
import torch
import numpy as np
import pickle as pkl
import kaolin as kal
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import point_to_surface, laplacian_loss  # , chamfer_distance
from kaolin.conversions import trianglemesh_to_sdf
from kaolin.rep import SDF as sdf
from psbody.mesh import Mesh, MeshViewer, MeshViewers
from tqdm import tqdm

from fit_SMPLX import save_meshes, backward_step
from fit_SMPLXD import optimize_offsets
from post_mesh_fix import optimize_offsets_only
# from fit_SMPLXD import forward_step as forward_step_offsets
# from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.th_SMPLX import th_batch_SMPLX, th_batch_SMPLX_split_params
from lib.mesh_distance import chamfer_distance, batch_point_to_surface
from lib.body_objectives import batch_get_pose_obj
from lib.smplx.body_models import SMPLX, SMPL
NUM_PARTS = 14  # number of parts that the smplx is segmented into.

# hands_vert = pkl.load(open('/home/chen/Downloads/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl', 'rb'), encoding='latin1')
# left_hand_vert = hands_vert['left_hand']
# right_hand_vert = hands_vert['right_hand']

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
                   'm2s': lambda cst, it: 10. ** 2 * cst / (1 + it),
                   'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                   # 'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                   'lap': lambda cst, it: cst / (1 + it),
                   'part': lambda cst, it: 10. ** 2 * cst / (1 + it),
                   'interpenetration': lambda cst, it: 10. ** -5 * cst/ (1 + it)
                   }
    return loss_weight


def forward_step(th_scan_meshes, smplx, scan_part_labels, smplx_part_labels, search_tree=None, pen_distance=None, tri_filtering_module=None):
    """
    Performs a forward step, given smplx and scan meshes.
    Then computes the losses.
    """
    # Get pose prior
    prior = get_prior(smplx.gender, precomputed=True)

    # forward
    # verts, _, _, _ = smplx()
    verts = smplx()
    th_smplx_meshes = [tm.from_tensors(vertices=v,
                                       faces=smplx.faces) for v in verts]

    scan_verts = [sm.vertices for sm in th_scan_meshes]
    smplx_verts = [sm.vertices for sm in th_smplx_meshes]

    # losses
    loss = dict()
    loss['s2m'] = batch_point_to_surface(scan_verts, th_smplx_meshes)
    loss['m2s'] = batch_point_to_surface(smplx_verts, th_scan_meshes)
    loss['betas'] = torch.mean(smplx.betas ** 2, axis=1)
    # loss['pose_pr'] = prior(smplx.pose)
    loss['interpenetration'] = interpenetration_loss(verts, smplx.faces, search_tree, pen_distance, tri_filtering_module, 1.0)
    loss['part'] = []
    for n, (sc_v, sc_l) in enumerate(zip(scan_verts, scan_part_labels)):
        tot = 0
        for i in range(NUM_PARTS):  # we currently use 14 parts
            if i not in sc_l:
                continue
            ind = torch.where(sc_l == i)[0]
            sc_part_points = sc_v[ind].unsqueeze(0)
            sm_part_points = smplx_verts[n][torch.where(smplx_part_labels[n] == i)[0]].unsqueeze(0)
            dist = chamfer_distance(sc_part_points, sm_part_points, w1=1., w2=1.)
            tot += dist
        loss['part'].append(tot / NUM_PARTS)
    loss['part'] = torch.stack(loss['part'])
    return loss


def optimize_pose_shape(th_scan_meshes, smplx, iterations, steps_per_iter, scan_part_labels, smplx_part_labels, search_tree=None, pen_distance=None, tri_filtering_module=None,
                        display=None):
    """
    Optimize SMPLX.
    :param display: if not None, pass index of the scan in th_scan_meshes to visualize.
    """
    # smplx.expression.requires_grad = False
    # smplx.jaw_pose.requires_grad = False
    # Optimizer
    optimizer = torch.optim.Adam([smplx.trans, smplx.betas, smplx.global_pose, smplx.body_pose,
                                  smplx.left_hand_pose, smplx.right_hand_pose], 0.02, betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    # Display
    if display is not None:
        assert int(display) < len(th_scan_meshes)
        mv = MeshViewer()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPLX')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, smplx, scan_part_labels, smplx_part_labels, search_tree, pen_distance, tri_filtering_module)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

            if display is not None:
                # verts, _, _, _ = smplx()
                verts = smplx()
                smplx_mesh = Mesh(v=verts[display].cpu().detach().numpy(), f=smplx.faces.cpu().numpy())
                scan_mesh = Mesh(v=th_scan_meshes[display].vertices.cpu().detach().numpy(),
                                 f=th_scan_meshes[display].faces.cpu().numpy(), vc=np.array([0, 1, 0]))
                scan_mesh.set_vertex_colors_from_weights(scan_part_labels[display].cpu().detach().numpy())
                mv.set_static_meshes([scan_mesh, smplx_mesh])

    print('** Optimised smplx pose and shape **')


def optimize_pose_only(th_scan_meshes, smplx, iterations, steps_per_iter, scan_part_labels, smplx_part_labels, search_tree=None, pen_distance=None, tri_filtering_module=None,
                       display=None):
    """
    Initially we want to only optimize the global rotation of SMPLX. Next we optimize full pose.
    We optimize pose based on the 3D keypoints in th_pose_3d.
    :param  th_pose_3d: array containing the 3D keypoints.
    """

    batch_sz = 1# smplx.pose.shape[0]
    split_smplx = th_batch_SMPLX_split_params(batch_sz, top_betas=smplx.betas.data[:, :2], other_betas=smplx.betas.data[:, 2:],
                                              global_pose=smplx.global_pose.data, body_pose=smplx.body_pose.data,
                                              left_hand_pose=smplx.left_hand_pose.data, right_hand_pose=smplx.right_hand_pose.data, 
                                              expression=smplx.expression.data, jaw_pose=smplx.jaw_pose.data, 
                                              leye_pose=smplx.leye_pose.data, reye_pose=smplx.reye_pose.data,
                                              faces=smplx.faces, gender=smplx.gender).to(DEVICE)
    # split_smplx.expression.requires_grad = False
    # split_smplx.jaw_pose.requires_grad = False
    optimizer = torch.optim.Adam([split_smplx.trans, split_smplx.top_betas, split_smplx.global_pose], 0.02,
                                 betas=(0.9, 0.999))

    # Get loss_weights
    weight_dict = get_loss_weights()

    if display is not None:
        assert int(display) < len(th_scan_meshes)
        # mvs = MeshViewers((1,1))
        mv = MeshViewer(keepalive=True)

    iter_for_global = 1
    for it in range(iter_for_global + iterations):
        loop = tqdm(range(steps_per_iter))
        if it < iter_for_global:
            # Optimize global orientation
            print('Optimizing SMPLX global orientation')
            loop.set_description('Optimizing SMPLX global orientation')
        elif it == iter_for_global:
            # Now optimize full SMPLX pose
            print('Optimizing SMPLX pose only')
            loop.set_description('Optimizing SMPLX pose only')
            optimizer = torch.optim.Adam([split_smplx.trans, split_smplx.top_betas, split_smplx.global_pose,
                                          split_smplx.body_pose, split_smplx.left_hand_pose, split_smplx.right_hand_pose], 0.02, betas=(0.9, 0.999))
        else:
            loop.set_description('Optimizing SMPLX pose only')

        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_scan_meshes, split_smplx, scan_part_labels, smplx_part_labels, search_tree, pen_distance, tri_filtering_module)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                loop.set_description(l_str)

            if display is not None:
                # verts, _, _, _ = split_smplx()
                verts = split_smplx()
                smplx_mesh = Mesh(v=verts[display].cpu().detach().numpy(), f=smplx.faces.cpu().numpy())
                scan_mesh = Mesh(v=th_scan_meshes[display].vertices.cpu().detach().numpy(),
                                 f=th_scan_meshes[display].faces.cpu().numpy(), vc=np.array([0, 1, 0]))
                scan_mesh.set_vertex_colors_from_weights(scan_part_labels[display].cpu().detach().numpy())

                mv.set_dynamic_meshes([smplx_mesh, scan_mesh])

    # Put back pose, shape and trans into original smplx
    smplx.global_pose.data = split_smplx.global_pose.data
    smplx.body_pose.data = split_smplx.body_pose.data
    smplx.left_hand_pose.data = split_smplx.left_hand_pose.data
    smplx.right_hand_pose.data = split_smplx.right_hand_pose.data
    # smplx.jaw_pose.data = split_smplx.jaw_pose.data
    smplx.leye_pose.data = split_smplx.leye_pose.data
    smplx.reye_pose.data = split_smplx.reye_pose.data
    smplx.betas.data = split_smplx.betas.data
    smplx.trans.data = split_smplx.trans.data

    print('** Optimised smplx pose **')


def fit_SMPLX(scans, scan_labels, gender='male', save_path=None, scale_file=None, display=None, interpenetration = True):
    """
    :param save_path:
    :param scans: list of scan paths
    :param pose_files:
    :return:
    """
    search_tree = None
    pen_distance = None
    tri_filtering_module = None
    max_collisions = 128
    df_cone_height = 0.0001
    point2plane=False
    penalize_outside=True
    part_segm_fn = '/home/chen/IPNet_SMPLX/assets/smplx_parts_segm.pkl'
    ign_part_pairs = ["9,16", "9,17", "6,16", "6,17", "1,2", "12,22"]
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces
        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=df_cone_height, point2plane=point2plane,
                                                                    vectorized=True, penalize_outside=penalize_outside)
        if part_segm_fn:
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pkl.load(faces_parents_file,encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            tri_filtering_module = FilterFaces(faces_segm=faces_segm, faces_parents=faces_parents,
                                               ign_part_pairs=ign_part_pairs).cuda()
    
    # Get SMPLX faces
    # spx = SmplPaths(gender=gender)
    spx = SMPLX(model_path="/home/chen/SMPLX/models/smplx", batch_size=1, gender=gender)
    smplx_faces = spx.faces
    th_faces = torch.tensor(smplx_faces.astype('float32'), dtype=torch.long).to(DEVICE)

    # Load SMPLX parts
    part_labels = pkl.load(open('/home/chen/IPNet_SMPLX/assets/smplx_parts_dense.pkl', 'rb'))
    labels = np.zeros((10475,), dtype='int32')
    for n, k in enumerate(part_labels):
        labels[part_labels[k]] = n
    labels = torch.tensor(labels).unsqueeze(0).to(DEVICE)

    # Load scan parts
    scan_part_labels = []
    for sc_l in scan_labels:
        temp = torch.tensor(np.load(sc_l).astype('int32')).to(DEVICE)
        scan_part_labels.append(temp)

    # Batch size
    batch_sz = len(scans)

    # Set optimization hyper parameters
    iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 3, 2, 30, 30

    # prior = get_prior(gender=gender, precomputed=True)
    if gender == 'male':
        temp_model = pkl.load(open('/home/chen/SMPLX/models/smplx/SMPLX_MALE.pkl', 'rb'), encoding='latin1')
    elif gender =='female':
        temp_model = pkl.load(open('/home/chen/SMPLX/models/smplx/SMPLX_FEMALE.pkl', 'rb'), encoding='latin1')
    else:
        print('Wrong gender input!')
        exit()
    left_hand_mean = torch.tensor(temp_model['hands_meanl']).unsqueeze(0)
    right_hand_mean = torch.tensor(temp_model['hands_meanr']).unsqueeze(0)
    # pose_init = torch.zeros((batch_sz, 69))
    # TODO consider to add the prior for smplx
    # pose_init[:, 3:] = prior.mean
    # betas, pose, trans = torch.zeros((batch_sz, 300)), pose_init, torch.zeros((batch_sz, 3))
    betas, global_pose, body_pose, trans = torch.zeros((batch_sz, 10)), torch.zeros((batch_sz, 3)), torch.zeros((batch_sz, 63)), torch.zeros((batch_sz, 3))
    left_hand_pose, right_hand_pose, expression, jaw_pose = left_hand_mean, right_hand_mean, torch.zeros((batch_sz, 10)), torch.zeros((batch_sz, 3))
    leye_pose, reye_pose = torch.zeros((batch_sz, 3)), torch.zeros((batch_sz, 3))
    # Init SMPLX, pose with mean smplx pose, as in ch.registration
    smplx = th_batch_SMPLX(batch_sz, betas, global_pose, body_pose, left_hand_pose, right_hand_pose, trans, 
                           expression, jaw_pose, leye_pose, reye_pose, faces=th_faces, gender=gender).to(DEVICE)
    smplx_part_labels = torch.cat([labels] * batch_sz, axis=0)

    th_scan_meshes, centers = [], []
    for scan in scans:
        print('scan path ...', scan)
        temp = Mesh(filename=scan)
        th_scan = tm.from_tensors(torch.tensor(temp.v.astype('float32'), requires_grad=False, device=DEVICE),
                                  torch.tensor(temp.f.astype('int32'), requires_grad=False, device=DEVICE).long())
        th_scan_meshes.append(th_scan)

    if scale_file is not None:
        for n, sc in enumerate(scale_file):
            dat = np.load(sc, allow_pickle=True)
            th_scan_meshes[n].vertices += torch.tensor(dat[1]).to(DEVICE)
            th_scan_meshes[n].vertices *= torch.tensor(dat[0]).to(DEVICE)

    # Optimize pose first
    optimize_pose_only(th_scan_meshes, smplx, pose_iterations, pose_steps_per_iter, scan_part_labels, smplx_part_labels, search_tree, pen_distance, tri_filtering_module,
                       display=None if display is None else 0)

    # Optimize pose and shape
    optimize_pose_shape(th_scan_meshes, smplx, iterations, steps_per_iter, scan_part_labels, smplx_part_labels, search_tree, pen_distance, tri_filtering_module,
                        display=None if display is None else 0)

    # verts, _, _, _ = smplx()
    verts = smplx()
    th_smplx_meshes = [tm.from_tensors(vertices=v, faces=smplx.faces) for v in verts]

    if save_path is not None:
        if not exists(save_path):
            os.makedirs(save_path)

        names = [split(s)[1] for s in scans]

        # Save meshes
        save_meshes(th_smplx_meshes, [join(save_path, n.replace('.ply', '_smplx.obj')) for n in names])
        save_meshes(th_scan_meshes, [join(save_path, n) for n in names])

        # Save params
        for g, bp, lh, rh, e, j, le, re, b, t, n in zip(smplx.global_pose.cpu().detach().numpy(), smplx.body_pose.cpu().detach().numpy(),
                                                        smplx.left_hand_pose.cpu().detach().numpy(), smplx.right_hand_pose.cpu().detach().numpy(),
                                                        smplx.expression.cpu().detach().numpy(), smplx.jaw_pose.cpu().detach().numpy(),
                                                        smplx.leye_pose.cpu().detach().numpy(), smplx.reye_pose.cpu().detach().numpy(),
                                                        smplx.betas.cpu().detach().numpy(),
                                                        smplx.trans.cpu().detach().numpy(), names):
            smplx_dict = {'global_pose': g, 'body_pose': bp, 'left_hand_pose': lh, 'right_hand_pose':rh,
                          'expression': e, 'jaw_pose': j, 'leye_pose': le, 'reye_pose': re,
                          'betas': b, 'trans': t}
            pkl.dump(smplx_dict, open(join(save_path, n.replace('.ply', '_smplx.pkl')), 'wb'))

        return (smplx.global_pose.cpu().detach().numpy(), smplx.body_pose.cpu().detach().numpy(),
                smplx.left_hand_pose.cpu().detach().numpy(), smplx.right_hand_pose.cpu().detach().numpy(),
                smplx.expression.cpu().detach().numpy(), smplx.jaw_pose.cpu().detach().numpy(),
                smplx.leye_pose.cpu().detach().numpy(), smplx.reye_pose.cpu().detach().numpy(),
                smplx.betas.cpu().detach().numpy(),
                smplx.trans.cpu().detach().numpy())


def fit_SMPLXD(scans, smplx_pkl, gender='male', save_path=None, scale_file=None, interpenetration = True):

    search_tree = None
    pen_distance = None
    tri_filtering_module = None
    max_collisions = 128
    df_cone_height = 0.0001
    point2plane=False
    penalize_outside=True
    part_segm_fn = '/home/chen/IPNet_SMPLX/assets/smplx_parts_segm.pkl'
    ign_part_pairs = ["9,16", "9,17", "6,16", "6,17", "1,2", "12,22"]
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces
        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=df_cone_height, point2plane=point2plane,
                                                                    vectorized=True, penalize_outside=penalize_outside)
        if part_segm_fn:
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pkl.load(faces_parents_file,encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            tri_filtering_module = FilterFaces(faces_segm=faces_segm, faces_parents=faces_parents,
                                               ign_part_pairs=ign_part_pairs).cuda()

    # Get SMPLX faces
    # spx = SmplPaths(gender=gender)
    spx = SMPLX(model_path="/home/chen/SMPLX/models/smplx", batch_size=1, gender=gender)
    smplx_faces = spx.faces
    th_faces = torch.tensor(smplx_faces.astype('float32'), dtype=torch.long).cuda()

    # Batch size
    batch_sz = len(scans)

    # Init SMPLX
    global_pose, body_pose, left_hand_pose, right_hand_pose = [], [], [], []
    expression, jaw_pose, leye_pose, reye_pose = [], [], [], []
    betas, trans = [], []
    for spkl in smplx_pkl:
        smplx_dict = pkl.load(open(spkl, 'rb'))
        g, bp, lh, rh, e, j, le, re, b, t = (smplx_dict['global_pose'], smplx_dict['body_pose'],
                                             smplx_dict['left_hand_pose'], smplx_dict['right_hand_pose'],
                                             smplx_dict['expression'], smplx_dict['jaw_pose'],
                                             smplx_dict['leye_pose'], smplx_dict['reye_pose'],        
                                             smplx_dict['betas'], smplx_dict['trans'])
        global_pose.append(g)
        body_pose.append(bp)
        left_hand_pose.append(lh)
        right_hand_pose.append(rh)
        expression.append(e)
        jaw_pose.append(j)
        leye_pose.append(le)
        reye_pose.append(re)
        if len(b) == 10:
            # temp = np.zeros((300,))
            temp = np.zeros((10,))
            temp[:10] = b
            b = temp.astype('float32')
        betas.append(b)
        trans.append(t)
    global_pose, body_pose, left_hand_pose, right_hand_pose = np.array(global_pose), np.array(body_pose), \
                                                              np.array(left_hand_pose), np.array(right_hand_pose)
    expression, jaw_pose, leye_pose, reye_pose = np.array(expression), np.array(jaw_pose), \
                                                 np.array(leye_pose), np.array(reye_pose)
    betas, trans = np.array(betas), np.array(trans)

    global_pose, body_pose, left_hand_pose, right_hand_pose = torch.tensor(global_pose), torch.tensor(body_pose), \
                                                              torch.tensor(left_hand_pose), torch.tensor(right_hand_pose)
    expression, jaw_pose, leye_pose, reye_pose = torch.tensor(expression), torch.tensor(jaw_pose), \
                                                 torch.tensor(leye_pose), torch.tensor(reye_pose)
    betas, trans = torch.tensor(betas), torch.tensor(trans)
    # smplx = th_batch_SMPLX(batch_sz, betas, pose, trans, faces=th_faces, gender=gender).cuda()
    smplx = th_batch_SMPLX(batch_sz, betas, global_pose, body_pose, left_hand_pose, right_hand_pose, trans, 
                           expression, jaw_pose, leye_pose, reye_pose, faces=th_faces, gender=gender).to(DEVICE)
    # verts, _, _, _ = smplx()
    verts = smplx()
    init_smplx_meshes = [tm.from_tensors(vertices=v.clone().detach(),
                                         faces=smplx.faces) for v in verts]

    # Load scans
    th_scan_meshes = []
    for scan in scans:
        print('scan path ...', scan)
        temp = Mesh(filename=scan)
        th_scan = tm.from_tensors(torch.tensor(temp.v.astype('float32'), requires_grad=False, device=DEVICE),
                                  torch.tensor(temp.f.astype('int32'), requires_grad=False, device=DEVICE).long())
        th_scan_meshes.append(th_scan)

    if scale_file is not None:
        for n, sc in enumerate(scale_file):
            dat = np.load(sc, allow_pickle=True)
            th_scan_meshes[n].vertices += torch.tensor(dat[1]).to(DEVICE)
            th_scan_meshes[n].vertices *= torch.tensor(dat[0]).to(DEVICE)

    # Optimize
    optimize_offsets(th_scan_meshes, smplx, init_smplx_meshes, 5, 10, search_tree, pen_distance, tri_filtering_module)
    # optimize_offsets_only(th_scan_meshes, smplx, init_smplx_meshes, 5, 8, search_tree, pen_distance, tri_filtering_module)
    print('Done')

    # verts, _, _, _ = smplx()
    verts = smplx.get_vertices_clean_hand()
    th_smplx_meshes = [tm.from_tensors(vertices=v,
                                       faces=smplx.faces) for v in verts]

    if save_path is not None:
        if not exists(save_path):
            os.makedirs(save_path)

        names = ['full.ply'] # [split(s)[1] for s in scans]

        # Save meshes
        save_meshes(th_smplx_meshes, [join(save_path, n.replace('.ply', '_smplxd.obj')) for n in names])
        save_meshes(th_scan_meshes, [join(save_path, n) for n in names])
        # Save params
        for g, bp, lh, rh, e, j, le, re, b, t, d, n in zip(smplx.global_pose.cpu().detach().numpy(), smplx.body_pose.cpu().detach().numpy(),
                                                           smplx.left_hand_pose.cpu().detach().numpy(), smplx.right_hand_pose.cpu().detach().numpy(),
                                                           smplx.expression.cpu().detach().numpy(), smplx.jaw_pose.cpu().detach().numpy(),
                                                           smplx.leye_pose.cpu().detach().numpy(), smplx.reye_pose.cpu().detach().numpy(),
                                                           smplx.betas.cpu().detach().numpy(),
                                                           smplx.trans.cpu().detach().numpy(), smplx.offsets_clean_hand.cpu().detach().numpy(), names):
            smplx_dict = {'global_pose': g, 'body_pose': bp, 'left_hand_pose': lh, 'right_hand_pose':rh,
                          'expression': e, 'jaw_pose': j, 'leye_pose': le, 'reye_pose': re,
                          'betas': b, 'trans': t, 'offsets': d}
            pkl.dump(smplx_dict, open(join(save_path, n.replace('.ply', '_smplxd.pkl')), 'wb'))

    return (smplx.global_pose.cpu().detach().numpy(), smplx.body_pose.cpu().detach().numpy(),
            smplx.left_hand_pose.cpu().detach().numpy(), smplx.right_hand_pose.cpu().detach().numpy(),
            smplx.expression.cpu().detach().numpy(), smplx.jaw_pose.cpu().detach().numpy(),
            smplx.leye_pose.cpu().detach().numpy(), smplx.reye_pose.cpu().detach().numpy(),
            smplx.betas.cpu().detach().numpy(),
            smplx.trans.cpu().detach().numpy(),
            smplx.offsets_clean_hand.cpu().detach().numpy())

DEVICE = 'cuda'
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('inner_path', type=str)  # predicted by IPNet
    # parser.add_argument('outer_path', type=str)  # predicted by IPNet
    parser.add_argument('inner_labels', type=str)  # predicted by IPNet
    parser.add_argument('scale_file', type=str, default=None)  # obtained from utils/process_scan.py
    parser.add_argument('save_path', type=str)
    # parser.add_argument('-gender', type=str, default='female')  # can be female/ male/ neutral
    parser.add_argument('--display', default=None)
    args = parser.parse_args()

    # args = lambda: None
    # args.inner_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/body.ply'
    # args.outer_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/full.ply'
    # args.inner_labels = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/parts.npy'
    # args.scale_file = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data/cent.npy'
    # args.display = None
    # args.save_path = '/BS/bharat-3/work/IPNet/DO_NOT_RELEASE/test_data'
    # args.gender = 'male'
    
    # args.outer_path = '/home/chen/IPNet/out_dir2/full.ply'
    # args.gender = 'male'
    args.outer_path = '/home/chen/IPNet_SMPLX/assets/pifu_golf_scaled.obj'
    args.gender = 'female'
    # fit_SMPLX([args.inner_path], scan_labels=[args.inner_labels], display=args.display, save_path=args.save_path,
    #           scale_file=[args.scale_file], gender=args.gender)

    names = [split(s)[1] for s in [args.inner_path]]
    smplx_pkl = [join(args.save_path, n.replace('.ply', '_smplx.pkl')) for n in names]

    fit_SMPLXD([args.outer_path], smplx_pkl=smplx_pkl, save_path=args.save_path,
               scale_file=[args.scale_file], gender=args.gender)
