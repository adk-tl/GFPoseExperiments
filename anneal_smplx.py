"""Generate an initial GF-Pose skeleton sample from a neutral SMPL-X body."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

import smplx
from lib.algorithms.advanced import sampling, sde_lib
from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.ema import ExponentialMovingAverage
from lib.dataset.h36m import normalize_data
from lib.utils.transforms import procrustes
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


N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3


def load_config_from_py(config_path: str):
    """Load a config object from a python config file exposing get_config()."""
    path = Path(config_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise ValueError(f"Config module {config_path} does not define get_config()")
    return module.get_config()


def load_score_and_ema(config, checkpoint_path: str, device: torch.device):
    """Initialize ScoreModelFC_Adv and EMA, then restore from checkpoint."""
    score_model = ScoreModelFC_Adv(
        config,
        n_joints=N_JOINTS,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        cond_dim=CONDITION_DIM,
    ).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    score_model.load_state_dict(checkpoint["model_state_dict"])
    ema.load_state_dict(checkpoint["ema"])
    return score_model, ema


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


def _build_sde(config, steps: int):
    """Build the SDE and epsilon used by sampler for a given step count."""
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        return (
            sde_lib.VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=steps,
                T=1.0,
            ),
            1e-3,
        )
    if sde_name == "subvpsde":
        return (
            sde_lib.subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=steps,
                T=1.0,
            ),
            1e-3,
        )
    if sde_name == "vesde":
        return (
            sde_lib.VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=steps,
                T=1.0,
            ),
            1e-5,
        )
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


def anneal_smplx_pose(
    smplx_angles: np.ndarray,
    smplx_model: smplx.SMPLX,
    score_model: ScoreModelFC_Adv,
    ema: ExponentialMovingAverage,
    steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and anneal an SMPL-X pose into GF-Pose joint space.

    Args:
        smplx_angles: SMPL-X pose angles shaped [165] or [B, 165], ordered as
            [global_orient(3), body_pose(63), jaw_pose(3),
             left_hand_pose(45), right_hand_pose(45), leye_pose(3), reye_pose(3)].
        smplx_model: Initialized SMPL-X model instance.
        score_model: Initialized and checkpoint-loaded score model.
        ema: EMA wrapper already loaded from checkpoint for score_model.
        steps: Number of annealing steps.

    Returns:
        (pre_anneal, post_anneal) as [B, 17, 3] normalized GF-Pose skeletons.
    """
    angles = np.asarray(smplx_angles, dtype=np.float32)
    if angles.ndim == 1:
        angles = angles[None, :]
    if angles.ndim != 2 or angles.shape[1] != 165:
        raise ValueError(f"Expected SMPL-X angles with shape [B, 165], got {angles.shape}")

    device = next(score_model.parameters()).device
    smplx_device = next(smplx_model.parameters()).device
    batch_size = angles.shape[0]
    angles_t = torch.tensor(angles, dtype=torch.float32, device=smplx_device)

    output = smplx_model(
        betas=torch.zeros(batch_size, 300, dtype=torch.float32, device=smplx_device),
        global_orient=angles_t[:, :3],
        body_pose=angles_t[:, 3:66],
        jaw_pose=angles_t[:, 66:69],
        left_hand_pose=angles_t[:, 69:114],
        right_hand_pose=angles_t[:, 114:159],
        leye_pose=angles_t[:, 159:162],
        reye_pose=angles_t[:, 162:165],
        expression=torch.zeros(batch_size, 10, dtype=torch.float32, device=smplx_device),
        return_verts=False,
    )

    pre_anneal = map_smplx_to_gfpose(output.joints.detach().cpu().numpy()).astype(np.float32)
    pre_anneal = normalize_data(pre_anneal).astype(np.float32)

    config = score_model.config
    data_scale = config.training.data_scale
    denoise_x = torch.tensor(pre_anneal, device=device) * data_scale
    condition = torch.zeros_like(denoise_x)

    sde, sampling_eps = _build_sde(config, steps)
    sampling_fn = sampling.get_sampling_fn(
        config,
        sde,
        (batch_size, 17, 3),
        inverse_scaler=lambda x: x,
        eps=sampling_eps,
        device=device,
    )

    score_model.eval()
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    try:
        with torch.no_grad():
            trajs, _ = sampling_fn(
                score_model,
                condition=condition,
                denoise_x=denoise_x,
                args=SimpleNamespace(task="den"),
            )
    finally:
        ema.restore(score_model.parameters())

    post_anneal = (trajs[-1] / data_scale).detach().cpu().numpy().astype(np.float32)
    for idx in range(batch_size):
        post_anneal[idx] = procrustes(pre_anneal[idx], post_anneal[idx], reflection=False)[1]

    return pre_anneal, post_anneal


def save_step0_npz(path: str, joints: np.ndarray) -> None:
    """Save joints under key step_0 for run/view_anneal_samples_open3d.py."""
    np.savez(path, step_0=joints.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anneal a zero-angle SMPL-X pose into GF-Pose space")
    parser.add_argument("--config", type=str, required=True, help="Path to a python config with get_config().")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Checkpoint path with model and EMA states.")
    parser.add_argument("--steps", type=int, default=10, help="Annealing steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config_from_py(args.config)
    smplx_model = load_smplx_model()
    score_model, ema = load_score_and_ema(config, args.ckpt_path, device=device)

    zero_angles = np.zeros(165, dtype=np.float32)
    pre_anneal, post_anneal = anneal_smplx_pose(
        smplx_angles=zero_angles,
        smplx_model=smplx_model,
        score_model=score_model,
        ema=ema,
        steps=args.steps,
    )

    np.savez("pose_test.npz", pre_anneal=pre_anneal.astype(np.float32), post_anneal=post_anneal.astype(np.float32))
    print("Saved pre/post annealed poses to pose_test.npz under keys 'pre_anneal' and 'post_anneal'.")


if __name__ == "__main__":
    main()
