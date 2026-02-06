"""Open3D viewer for annealed pose samples saved by run.sample_anneal_from_ckpt."""

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import open3d as o3d

from lib.dataset.h36m import H36MDataset3D


@dataclass
class PoseSequence:
    poses: List[np.ndarray]
    keys: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View annealed pose samples in Open3D")
    parser.add_argument("--input", type=str, required=True, help="Path to .npz file")
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Which batch element to visualize (default: 0)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=6.0,
        help="Point size for joints",
    )
    return parser.parse_args()


def preprocess_pose(pose: np.ndarray) -> np.ndarray:
    """Match the axis convention used in the authors' viewer."""
    pose_3d = pose[:, [0, 2, 1]].copy()
    pose_3d[:, 2] *= -1
    return pose_3d


def load_pose_sequence(path: str, batch_index: int) -> PoseSequence:
    data_dict: Dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for key in data.files:
            data_dict[key] = data[key]

    step_items = []
    for key, value in data_dict.items():
        if not key.startswith("step_"):
            continue
        step_idx = int(key.split("step_")[-1])
        step_items.append((step_idx, key, value))

    if not step_items:
        raise ValueError("No step_* arrays found in the input file.")

    step_items.sort(key=lambda item: item[0])

    poses = []
    keys = []
    for _, key, value in step_items:
        if value.ndim != 3:
            raise ValueError(f"Expected [batch, joints, 3] arrays, got {value.shape}.")
        if not (0 <= batch_index < value.shape[0]):
            raise IndexError(
                f"batch-index {batch_index} out of range for {key} with batch {value.shape[0]}"
            )
        poses.append(preprocess_pose(value[batch_index]))
        keys.append(key)

    return PoseSequence(poses=poses, keys=keys)


class PoseViewer:
    def __init__(self, sequence: PoseSequence, skeleton: List[List[int]], point_size: float):
        self.sequence = sequence
        self.skeleton = skeleton
        self.point_size = point_size
        self.index = 0

        self.points = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self.lines.lines = o3d.utility.Vector2iVector(np.asarray(self.skeleton, dtype=np.int32))
        self.lines.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.1, 0.8, 0.1]]), (len(self.skeleton), 1))
        )

        self._apply_pose(self.sequence.poses[self.index])

    def _apply_pose(self, pose: np.ndarray) -> None:
        self.points.points = o3d.utility.Vector3dVector(pose)
        self.points.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.9, 0.1, 0.1]]), (pose.shape[0], 1))
        )
        self.lines.points = o3d.utility.Vector3dVector(pose)

    def _load_pose(self, vis: o3d.visualization.Visualizer, new_index: int) -> None:
        new_index = max(0, min(new_index, len(self.sequence.poses) - 1))
        if new_index == self.index:
            return
        self.index = new_index
        self._apply_pose(self.sequence.poses[self.index])
        vis.update_geometry(self.points)
        vis.update_geometry(self.lines)
        vis.update_renderer()
        print(f"Showing {self.sequence.keys[self.index]} ({self.index + 1}/{len(self.sequence.poses)})")

    def next_pose(self, vis: o3d.visualization.Visualizer) -> bool:
        self._load_pose(vis, self.index + 1)
        return False

    def prev_pose(self, vis: o3d.visualization.Visualizer) -> bool:
        self._load_pose(vis, self.index - 1)
        return False

    def show(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Pose Sample Viewer")
        vis.add_geometry(self.points)
        vis.add_geometry(self.lines)

        render_option = vis.get_render_option()
        render_option.point_size = self.point_size
        render_option.line_width = 2.0

        vis.register_key_callback(ord("-"), self.prev_pose)
        vis.register_key_callback(ord("="), self.next_pose)

        print("Controls: '-' previous pose, '=' next pose, standard Open3D controls for navigation.")
        print(f"Showing {self.sequence.keys[self.index]} ({self.index + 1}/{len(self.sequence.poses)})")
        vis.run()
        vis.destroy_window()


def main() -> None:
    args = parse_args()
    sequence = load_pose_sequence(args.input, args.batch_index)
    skeleton = H36MDataset3D.get_skeleton()
    viewer = PoseViewer(sequence, skeleton, point_size=args.point_size)
    viewer.show()


if __name__ == "__main__":
    main()
