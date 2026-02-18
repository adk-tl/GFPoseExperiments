"""Visualize original vs annealed pose from a .npz file in Open3D.

Defaults to reading ./pose_test.npz and rendering:
- original pose in red
- annealed pose in green
"""

import argparse
from typing import Dict, Iterable, List, Tuple

import numpy as np
import open3d as o3d

from lib.dataset.h36m import H36MDataset3D


ORIGINAL_CANDIDATES: Tuple[str, ...] = (
    "original",
    "orig",
    "input",
    "source",
    "pose_original",
    "pose_orig",
)
ANNEALED_CANDIDATES: Tuple[str, ...] = (
    "annealed",
    "output",
    "result",
    "pose_annealed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render two poses from an .npz file in Open3D")
    parser.add_argument(
        "--input",
        type=str,
        default="./pose_test.npz",
        help="Path to the .npz file (default: ./pose_test.npz)",
    )
    parser.add_argument(
        "--original-key",
        type=str,
        default=None,
        help="Array key for the original pose; auto-detected if omitted",
    )
    parser.add_argument(
        "--annealed-key",
        type=str,
        default=None,
        help="Array key for the annealed pose; auto-detected if omitted",
    )
    parser.add_argument("--point-size", type=float, default=7.0, help="Point size")
    return parser.parse_args()


def preprocess_pose(pose: np.ndarray) -> np.ndarray:
    """Match the axis convention used by the project's viewers."""
    pose_3d = pose[:, [0, 2, 1]].copy()
    pose_3d[:, 2] *= -1
    return pose_3d


def _pick_2d_pose(arr: np.ndarray, key: str) -> np.ndarray:
    """Collapse leading dimensions by selecting index 0 until [joints, 3]."""
    pose = arr
    while pose.ndim > 2:
        if pose.shape[0] == 0:
            raise ValueError(f"Array '{key}' has an empty leading dimension: {arr.shape}")
        pose = pose[0]

    if pose.ndim != 2 or pose.shape[-1] != 3:
        raise ValueError(f"Array '{key}' must reduce to [joints, 3], got {arr.shape} -> {pose.shape}")
    return pose


def _find_key(data: Dict[str, np.ndarray], requested: str, candidates: Iterable[str], label: str) -> str:
    if requested is not None:
        if requested not in data:
            raise KeyError(f"Requested {label} key '{requested}' not found. Available keys: {list(data.keys())}")
        return requested

    lower_to_key = {k.lower(): k for k in data.keys()}
    for candidate in candidates:
        if candidate in lower_to_key:
            return lower_to_key[candidate]

    if len(data) >= 2:
        return list(data.keys())[0 if label == "original" else 1]

    raise KeyError(f"Could not auto-detect {label} key. Available keys: {list(data.keys())}")


def load_poses(path: str, original_key: str = None, annealed_key: str = None) -> Tuple[np.ndarray, np.ndarray, str, str]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}

    if not arrays:
        raise ValueError(f"No arrays found in '{path}'")

    orig_key = _find_key(arrays, original_key, ORIGINAL_CANDIDATES, "original")
    ann_key = _find_key(arrays, annealed_key, ANNEALED_CANDIDATES, "annealed")
    if orig_key == ann_key:
        raise ValueError(
            f"Original key and annealed key resolved to the same array ('{orig_key}'). "
            "Pass --original-key and --annealed-key explicitly."
        )

    original = preprocess_pose(_pick_2d_pose(arrays[orig_key], orig_key))
    annealed = preprocess_pose(_pick_2d_pose(arrays[ann_key], ann_key))
    return original, annealed, orig_key, ann_key


def make_pose_geometry(pose: np.ndarray, skeleton: List[List[int]], color: np.ndarray):
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pose)
    points.colors = o3d.utility.Vector3dVector(np.tile(color[None, :], (pose.shape[0], 1)))

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(pose)
    lines.lines = o3d.utility.Vector2iVector(np.asarray(skeleton, dtype=np.int32))
    lines.colors = o3d.utility.Vector3dVector(np.tile(color[None, :], (len(skeleton), 1)))
    return points, lines


def main() -> None:
    args = parse_args()

    original, annealed, orig_key, ann_key = load_poses(
        args.input,
        original_key=args.original_key,
        annealed_key=args.annealed_key,
    )
    skeleton = H36MDataset3D.get_skeleton()

    orig_color = np.array([0.95, 0.2, 0.2], dtype=np.float64)
    annealed_color = np.array([0.2, 0.9, 0.2], dtype=np.float64)

    orig_points, orig_lines = make_pose_geometry(original, skeleton, orig_color)
    ann_points, ann_lines = make_pose_geometry(annealed, skeleton, annealed_color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="pose_test.npz viewer")
    vis.add_geometry(orig_points)
    vis.add_geometry(orig_lines)
    vis.add_geometry(ann_points)
    vis.add_geometry(ann_lines)

    render_option = vis.get_render_option()
    render_option.point_size = args.point_size
    render_option.line_width = 2.0

    print(f"Loaded original='{orig_key}' (red), annealed='{ann_key}' (green) from {args.input}")
    print("Use standard Open3D controls to orbit/pan/zoom.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
