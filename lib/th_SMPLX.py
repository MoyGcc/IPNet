'''
Takes in smplx parms and initialises a smplx object with optimizable params.
class th_SMPL currently does not take batch dim.
If code works:
    Author: Bharat
else:
    Author: Anonymous
'''
import numpy as np
import torch
import torch.nn as nn
from lib.smpl_layer import SMPL_Layer
from lib.body_objectives import torch_pose_obj_data
from lib.torch_functions import batch_sparse_dense_matmul
from lib.smplx.body_models import SMPLX
# binary_mask = torch.zeros((1, 10475, 3)).cuda()
# rest_idx = np.load('/home/chen/IPNet_SMPLX/assets/rest_idx.npy')
# binary_mask[:,rest_idx,:] = 1.
face_idx = np.load('/home/chen/Downloads/smplx_mano_flame_correspondences/SMPL-X__FLAME_vertex_ids.npy')
hand_idx = np.load('/home/chen/IPNet_SMPLX/assets/hand_idx.npy')
num_pca_comps=6
binary_mask = torch.ones((1, 10475, 3)).cuda()

binary_mask[:,face_idx,:] = 0.
binary_mask[:,hand_idx,:] = 0.
class th_batch_SMPLX_split_params(nn.Module):
    """
    Alternate implementation of th_batch_SMPL that allows us to independently optimise:
     1. global_pose
     2. remaining body_pose
     3. top betas (primarly adjusts bone lengths)
     4. other betas
    """
    def __init__(self, batch_sz=1, top_betas=None, other_betas=None, global_pose=None, body_pose=None, 
                 left_hand_pose=None, right_hand_pose=None, trans=None,
                 expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                 offsets=None, faces=None, gender='male'):
        super(th_batch_SMPLX_split_params, self).__init__()
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, 2))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, 8))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(batch_sz, 63))
        else:
            assert body_pose.ndim == 2
            self.body_pose = nn.Parameter(body_pose)
        if left_hand_pose is None:
            self.left_hand_pose = nn.Parameter(torch.zeros(batch_sz, 45))
        else:
            assert left_hand_pose.ndim == 2
            self.left_hand_pose = nn.Parameter(left_hand_pose)
        if right_hand_pose is None:
            self.right_hand_pose = nn.Parameter(torch.zeros(batch_sz, 45))
        else:
            assert right_hand_pose.ndim == 2
            self.right_hand_pose = nn.Parameter(right_hand_pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if expression is None:
            self.expression = nn.Parameter(torch.zeros(batch_sz, 10))
        else:
            assert expression.ndim == 2
            self.expression = nn.Parameter(expression)
        if jaw_pose is None:
            self.jaw_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert jaw_pose.ndim == 2
            self.jaw_pose = nn.Parameter(jaw_pose)
        if leye_pose is None:
            self.leye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert leye_pose.ndim == 2
            self.leye_pose = nn.Parameter(leye_pose)
        if reye_pose is None:
            self.reye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert reye_pose.ndim == 2
            self.reye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 10475, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        # self.pose = torch.cat([self.global_pose, self.body_pose], axis=1)

        self.faces = faces
        self.gender = gender
        # pytorch smplx
        self.smplx = SMPLX(model_path="/home/chen/SMPLX/models/smplx", batch_size=batch_sz, gender=gender) 

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

    def forward(self):
        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        # self.pose = torch.cat([self.global_pose, self.body_pose], axis=1)
        output = self.smplx(betas = self.betas,
                            global_orient = self.global_pose, 
                            body_pose = self.body_pose,
                            left_hand_pose = self.left_hand_pose[:, :num_pca_comps],
                            right_hand_pose = self.right_hand_pose[:, :num_pca_comps],
                            transl = self.trans,
                            expression = self.expression,
                            jaw_pose = self.jaw_pose,
                            leye_pose = self.leye_pose,
                            reye_pose = self.reye_pose,
                            displacement=self.offsets)
        # return verts, jtr, tposed, naked
        return output.vertices
    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smplx(self.pose,
                                    th_betas=self.betas,
                                    th_trans=self.trans,
                                    th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class th_batch_SMPLX(nn.Module):
    def __init__(self, batch_sz=1, betas=None, global_pose=None, body_pose=None, 
                 left_hand_pose=None, right_hand_pose=None, trans=None, 
                 expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                 offsets=None, faces=None, gender='male'):
        super(th_batch_SMPLX, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 10))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(batch_sz, 63))
        else:
            assert body_pose.ndim == 2
            self.body_pose = nn.Parameter(body_pose)
        if left_hand_pose is None:
            self.left_hand_pose = nn.Parameter(torch.zeros(batch_sz, 45))
        else:
            assert left_hand_pose.ndim == 2
            self.left_hand_pose = nn.Parameter(left_hand_pose)
        if right_hand_pose is None:
            self.right_hand_pose = nn.Parameter(torch.zeros(batch_sz, 45))
        else:
            assert right_hand_pose.ndim == 2
            self.right_hand_pose = nn.Parameter(right_hand_pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if expression is None:
            self.expression = nn.Parameter(torch.zeros(batch_sz, 10))
        else:
            assert expression.ndim == 2
            self.expression = nn.Parameter(expression)
        if jaw_pose is None:
            self.jaw_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert jaw_pose.ndim == 2
            self.jaw_pose = nn.Parameter(jaw_pose)
        if leye_pose is None:
            self.leye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert leye_pose.ndim == 2
            self.leye_pose = nn.Parameter(leye_pose)
        if reye_pose is None:
            self.reye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert reye_pose.ndim == 2
            self.reye_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 10475,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.faces = faces
        self.gender = gender
        # self.pose = torch.cat([self.global_pose, self.body_pose], axis=1)
        # pytorch smplx
        self.smplx = SMPLX(model_path="/home/chen/SMPLX/models/smplx", batch_size=batch_sz, gender=gender) 

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

    def forward(self):
        # self.pose = torch.cat([self.global_pose, self.body_pose], axis=1)
        output = self.smplx(betas = self.betas,
                            global_orient = self.global_pose, 
                            body_pose = self.body_pose,
                            left_hand_pose = self.left_hand_pose[:, :num_pca_comps],
                            right_hand_pose = self.right_hand_pose[:, :num_pca_comps],
                            transl = self.trans,
                            expression = self.expression,
                            jaw_pose = self.jaw_pose,
                            leye_pose = self.leye_pose,
                            reye_pose = self.reye_pose,
                            displacement=self.offsets)
        # return verts, jtr, tposed, naked
        return output.vertices
    def get_vertices_clean_hand(self):
        self.offsets_clean_hand = self.offsets.detach().clone()
        self.offsets_clean_hand = self.offsets_clean_hand * binary_mask
        output = self.smplx(betas = self.betas,
                            global_orient = self.global_pose, 
                            body_pose = self.body_pose,
                            left_hand_pose = self.left_hand_pose[:, :num_pca_comps],
                            right_hand_pose = self.right_hand_pose[:, :num_pca_comps],
                            transl = self.trans,
                            expression = self.expression,
                            jaw_pose = self.jaw_pose,
                            leye_pose = self.leye_pose,
                            reye_pose = self.reye_pose,
                            displacement=self.offsets_clean_hand)
        return output.vertices
    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smplx(self.pose,
                                    th_betas=self.betas,
                                    th_trans=self.trans,
                                    th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class th_SMPLX(nn.Module):
    def __init__(self, betas=None, global_pose=None, body_pose=None, 
                 left_hand_pose=None, right_hand_pose=None, trans=None, 
                 expression=None, jaw_pose=None, leye_pose=None, reye_pose=None, 
                 offsets=None, gender='male'):
        super(th_SMPLX, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(10,))
        else:
            self.betas = nn.Parameter(betas)
        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(3,))
        else:
            self.global_pose = nn.Parameter(global_pose)
        if body_pose is None:
            self.body_pose = nn.Parameter(torch.zeros(63,))
        else:
            self.body_pose = nn.Parameter(body_pose)
        if left_hand_pose is None:
            self.left_hand_pose = nn.Parameter(torch.zeros(45,))
        else:
            self.left_hand_pose = nn.Parameter(left_hand_pose)    
        if right_hand_pose is None:
            self.right_hand_pose = nn.Parameter(torch.zeros(45,))
        else:
            self.right_hand_pose = nn.Parameter(right_hand_pose)     
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if expression is None:
            self.expression = nn.Parameter(torch.zeros(10,))
        else:
            self.expression = nn.Parameter(expression)
        if jaw_pose is None:
            self.jaw_pose = nn.Parameter(torch.zeros(3,))
        else:
            self.jaw_pose = nn.Parameter(jaw_pose)
        if leye_pose is None:
            self.leye_pose = nn.Parameter(torch.zeros(3,))
        else:
            self.leye_pose = nn.Parameter(leye_pose)
        if reye_pose is None:
            self.reye_pose = nn.Parameter(torch.zeros(3,))
        else:
            self.reye_pose = nn.Parameter(reye_pose)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(10475,3))
        else:
            self.offsets = nn.Parameter(offsets)
        self.pose = torch.cat([self.global_pose, self.body_pose], axis=0)
        ## pytorch smplx
        self.smplx = SMPLX(model_path="/home/chen/SMPLX/models/smplx", batch_size=batch_size, gender=gender) 

    def forward(self):
        self.pose = torch.cat([self.global_pose, self.body_pose], axis=0)
        verts, jtr, tposed, naked = self.smplx(betas = self.betas.unsqueeze(axis=0),
                                               global_orient = self.global_pose.unsqueeze(axis=0), 
                                               body_pose = self.body_pose.unsqueeze(axis=0),
                                               left_hand_pose = self.left_hand_pose.unsqueeze(axis=0),
                                               right_hand_pose = self.right_hand_pose.unsqueeze(axis=0),
                                               transl = self.trans.unsqueeze(axis=0),
                                               expression = self.expression.unsqueeze(axis=0),
                                               jaw_pose = self.jaw_pose.unsqueeze(axis=0),
                                               leye_pose = self.leye_pose.unsqueeze(axis=0), 
                                               reye_pose = self.reye_pose.unsqueeze(axis=0),
                                               displacement=self.offsets.unsqueeze(axis=0))                                               
        return verts[0]
