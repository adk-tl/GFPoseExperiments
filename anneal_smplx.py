"""Generate an initial GF-Pose skeleton sample from a neutral SMPL-X body."""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

import smplx
from smpl_config import SMPLX_MODEL_PATH


# Map H36M 17-joint order used by this repository to SMPL-X body joints.
# H36M: [pelvis, rhip, rknee, rank, lhip, lknee, lank, spine, thorax,
#        neck, head, lsho, lelb, lwri, rsho, relb, rwri]
SMPLX_TO_H36M: Dict[str, int] = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

H36M_FROM_SMPLX_ORDER: List[int] = [
    SMPLX_TO_H36M["pelvis"],
    SMPLX_TO_H36M["right_hip"],
    SMPLX_TO_H36M["right_knee"],
    SMPLX_TO_H36M["right_ankle"],
    SMPLX_TO_H36M["left_hip"],
    SMPLX_TO_H36M["left_knee"],
    SMPLX_TO_H36M["left_ankle"],
    SMPLX_TO_H36M["spine1"],
    SMPLX_TO_H36M["spine3"],
    SMPLX_TO_H36M["neck"],
    SMPLX_TO_H36M["head"],
    SMPLX_TO_H36M["left_shoulder"],
    SMPLX_TO_H36M["left_elbow"],
    SMPLX_TO_H36M["left_wrist"],
    SMPLX_TO_H36M["right_shoulder"],
    SMPLX_TO_H36M["right_elbow"],
    SMPLX_TO_H36M["right_wrist"],
]


def load_smplx_model() -> smplx.SMPLX:
    """Load the SMPL-X model with all blend-shape coefficients enabled."""
    return smplx.create(
        model_path=SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=300,
        ext="npz",
    )


def map_smplx_to_gfpose(smplx_joints: np.ndarray) -> np.ndarray:
    """Map SMPL-X joints [B, J, 3] to GF-Pose/H36M order [B, 17, 3]."""
    if smplx_joints.ndim != 3 or smplx_joints.shape[-1] != 3:
        raise ValueError(f"Expected [batch, joints, 3], got {smplx_joints.shape}")
    return smplx_joints[:, H36M_FROM_SMPLX_ORDER, :]


def generate_step0_pose(model: smplx.SMPLX, batch_size: int = 1) -> np.ndarray:
    """Run SMPL-X with all-zero parameters and return GF-Pose-compatible joints."""
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32)

    output = model(
        betas=zeros(batch_size, 300),
        global_orient=zeros(batch_size, 3),
        body_pose=zeros(batch_size, 63),
        jaw_pose=zeros(batch_size, 3),
        left_hand_pose=zeros(batch_size, 45),
        right_hand_pose=zeros(batch_size, 45),
        leye_pose=zeros(batch_size, 3),
        reye_pose=zeros(batch_size, 3),
        expression=zeros(batch_size, 10),
        return_verts=False,
    )

    smplx_joints = output.joints.detach().cpu().numpy()
    return map_smplx_to_gfpose(smplx_joints)


def save_step0_npz(path: str, joints: np.ndarray) -> None:
    """Save joints under key step_0 for run/view_anneal_samples_open3d.py."""
    np.savez(path, step_0=joints.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a step_0 GF-Pose sample from SMPL-X")
    parser.add_argument("--output", type=str, default="anneal_smplx_step0.npz")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_smplx_model()
    joints = generate_step0_pose(model=model, batch_size=args.batch_size)
    save_step0_npz(args.output, joints)
    print(f"Saved {joints.shape} to {args.output} under key 'step_0'.")


if __name__ == "__main__":
    main()
