"""Visualization helpers for weighted quaternion samples."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from . import data_tools as tl
except ImportError:  # pragma: no cover - script execution fallback
    import data_tools as tl


def quat_rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Rotate a vector using a quaternion in ``xyzw`` format."""
    return R.from_quat(np.asarray(quaternion, dtype=np.float64)).apply(np.asarray(vector, dtype=np.float64))


def plot_quaternions_with_weights(quaternions: np.ndarray, weights: np.ndarray) -> None:
    """Visualize the rotated tool axis for each quaternion with a weight-based color map."""
    base_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotated_vectors = np.asarray([quat_rotate_vector(quaternion, base_vector) for quaternion in quaternions])
    average_quaternion = np.mean(quaternions, axis=0)
    average_quaternion /= np.linalg.norm(average_quaternion)
    average_vector = quat_rotate_vector(average_quaternion, base_vector)

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    min_weight = float(np.min(weights))
    max_weight = float(np.max(weights))
    scale = max(max_weight - min_weight, np.finfo(np.float32).eps)
    for rotated_vector, weight in zip(rotated_vectors, weights):
        normalized_weight = (float(weight) - min_weight) / scale
        color = plt.cm.summer(normalized_weight)
        axis.quiver(
            0.0,
            0.0,
            0.0,
            rotated_vector[0],
            rotated_vector[1],
            rotated_vector[2],
            color=color,
            length=1.0,
            normalize=True,
        )

    axis.quiver(
        0.0,
        0.0,
        0.0,
        average_vector[0],
        average_vector[1],
        average_vector[2],
        color="k",
        length=1.0,
        normalize=True,
        label="Average orientation",
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.grid(False)
    axis.set_axis_off()
    axis.view_init(elev=30, azim=45)
    axis.legend()
    plt.show()


def main() -> None:
    """Load the legacy demo dataset and render the quaternion visualization."""
    data_path = "./dataset/data.txt"
    joint_samples = tl.data_read(data_path)
    quaternions, _ = tl.data_process(joint_samples)
    weights, _ = tl.getWeightsforData(joint_samples)
    plot_quaternions_with_weights(quaternions, weights)


if __name__ == "__main__":
    main()
