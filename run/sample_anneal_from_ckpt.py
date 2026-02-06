"""Minimal script to sample a dataset pose and anneal with a denoising model."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

from lib.algorithms.advanced.model import ScoreModelFC_Adv
from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage
from lib.dataset.h36m import H36MDataset3D, denormalize_data, normalize_data

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False
)
flags.mark_flags_as_required(["config"])

N_JOINTS = 17
JOINT_DIM = 3
HIDDEN_DIM = 1024
EMBED_DIM = 512
CONDITION_DIM = 3


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description="anneal a dataset pose")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--std", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="anneal_samples.npz")
    parser.add_argument("--best", action="store_true")
    return parser.parse_args(argv[1:])


def build_sde(config, steps):
    """Build the SDE with the requested number of steps."""
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        return sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=steps,
            T=1.0,
        ), 1e-3
    if sde_name == "subvpsde":
        return sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=steps,
            T=1.0,
        ), 1e-3
    if sde_name == "vesde":
        return sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=steps,
            T=1.0,
        ), 1e-5
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")


def main(args):
    config = FLAGS.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the score model and restore weights.
    model = ScoreModelFC_Adv(
        config,
        n_joints=N_JOINTS,
        joint_dim=JOINT_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        cond_dim=CONDITION_DIM,
    ).to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, ema=ema, step=0)

    ckpt_name = "best_model.pth" if args.best else "checkpoint.pth"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    print(f"loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema.load_state_dict(checkpoint["ema"])
    state["step"] = checkpoint["step"]
    print(f"=> loaded checkpoint '{ckpt_path}' (step {state['step']})")

    model.eval()
    ema.copy_to(model.parameters())

    # Load poses from the dataset and add zero-mean Gaussian noise (std=10).
    dataset = H36MDataset3D(
        Path("data", "h36m"),
        "test",
        gt2d=True,
        read_confidence=False,
        sample_interval=None,
        flip=False,
        cond_3d_prob=0,
    )
    replace = args.batch > len(dataset.db_3d)
    indices = np.random.choice(len(dataset.db_3d), size=args.batch, replace=replace)
    base_pose = dataset.db_3d[indices]
    base_pose = denormalize_data(base_pose)
    noise = np.random.normal(loc=0.0, scale=args.std, size=base_pose.shape).astype(
        np.float32
    )
    noisy_pose = normalize_data(base_pose + noise).astype(np.float32)

    # Scale to the model space used during training.
    denoise_x = torch.tensor(noisy_pose, device=device) * config.training.data_scale
    condition = torch.zeros_like(denoise_x)

    # Setup the SDE and sampling function.
    sde, sampling_eps = build_sde(config, args.steps)
    sampling_shape = (args.batch, N_JOINTS, JOINT_DIM)
    sampling_fn = sampling.get_sampling_fn(
        config,
        sde,
        sampling_shape,
        inverse_scaler=lambda x: x,
        eps=sampling_eps,
        device=device,
    )

    # Minimal args namespace to signal denoising mode to the sampler.
    sampler_args = SimpleNamespace(task="den")

    with torch.no_grad():
        trajs, _ = sampling_fn(
            model,
            condition=condition,
            denoise_x=denoise_x,
            args=sampler_args,
        )

    # Convert back to unscaled coordinates and save each step with key `step_N`.
    trajs = trajs / config.training.data_scale
    save_dict = {f"step_{idx}": trajs[idx] for idx in range(trajs.shape[0])}
    np.savez(args.output, **save_dict)
    print(f"saved annealing trajectory to {args.output}")


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
