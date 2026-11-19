"""Vendored from MotionStreamer utils/eval_trans.py - ONLY the functions pami
uses (calculate_mpjpe / accumulate_rotations / recover_from_local_position / re-export
rotation_6d_to_matrix). The original module imports visualization.plot_3d_global at top level,
which we do not vendor - do not copy the full file."""
import numpy as np
import torch
from .face_z_align_util import rotation_6d_to_matrix


def calculate_mpjpe(gt_joints, pred_joints):
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) 
    mpjpe_seq = mpjpe.mean(-1)

    return mpjpe_seq


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):

    if final_x.ndim == 3:
        bs, nfrm, _ = final_x.shape
        is_batched = True
    else:
        nfrm, _ = final_x.shape
        bs = 1
        is_batched = False
        final_x = final_x.reshape(1, *final_x.shape)

    
    positions_no_heading = final_x[:,:,8:8+3*njoint].reshape(bs, nfrm, njoint, 3) 
    velocities_root_xy_no_heading = final_x[:,:,:2] 
    global_heading_diff_rot = final_x[:,:,2:8] 

   
    positions_with_heading = []
    for b in range(bs):
        
        global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot[b])).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        
        
        curr_pos_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), 
                                        positions_no_heading[b][...,None]).squeeze(-1)

        
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading[b].shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[b, :, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[b, :, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], 
                                                         velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

        
        curr_pos_with_heading[:, :, 0] += root_translation[:, 0:1]
        curr_pos_with_heading[:, :, 2] += root_translation[:, 2:]
        
        positions_with_heading.append(curr_pos_with_heading)

    positions_with_heading = np.stack(positions_with_heading, axis=0)

    if not is_batched:
        positions_with_heading = positions_with_heading.squeeze(0)

    return positions_with_heading



# Single-GPU evaluation of Causal TAE (test time)
