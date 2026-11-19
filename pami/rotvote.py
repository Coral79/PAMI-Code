"""Rotation helpers kept for pami/refine (the orientation-voting experiment itself was not used by the paper
checkpoints and has been removed)."""
import torch


def mat_to_rot6d(R):
    """(...,3,3) -> (...,6) row-6D (first two rows; the convention of vendor.eval_trans.rotation_6d_to_matrix)."""
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)
