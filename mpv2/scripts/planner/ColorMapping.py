#!/usr/bin/env python3
"""Color and point-cloud conversion helpers for ROS visualization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import rospy
from numpy import random as rnd
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def convert_numpy_2_pointcloud2_color(
    points: np.ndarray,
    ccc: np.ndarray,
    stamp=None,
    frame_id: str | None = None,
    maxDistColor: float | None = None,
) -> PointCloud2:
    """Create a colored ``PointCloud2`` message from point and color arrays."""
    dist = np.linalg.norm(points, axis=1)
    if maxDistColor is not None and maxDistColor > 0:
        dist = np.clip(dist, 0, maxDistColor)
    _ = dist

    rgba = np.zeros((len(points), 4), dtype=np.uint8) + 255
    rgba[:, 0] = (ccc[:, 2] * 255).astype(np.uint8)
    rgba[:, 1] = (ccc[:, 1] * 255).astype(np.uint8)
    rgba[:, 2] = (ccc[:, 0] * 255).astype(np.uint8)
    rgba = rgba.view("uint32")

    structured = np.zeros(
        (points.shape[0], 1),
        dtype={"names": ("x", "y", "z", "rgba"), "formats": ("f4", "f4", "f4", "u4")},
    )

    points = points.astype(np.float32)
    structured["x"] = points[:, 0].reshape((-1, 1))
    structured["y"] = points[:, 1].reshape((-1, 1))
    structured["z"] = points[:, 2].reshape((-1, 1))
    structured["rgba"] = rgba

    header = Header()
    header.stamp = rospy.Time().now() if stamp is None else stamp
    header.frame_id = "None" if frame_id is None else frame_id

    message = PointCloud2()
    message.header = header
    if len(points.shape) == 3:
        message.height = points.shape[1]
        message.width = points.shape[0]
    else:
        message.height = 1
        message.width = points.shape[0]

    message.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 12, PointField.UINT32, 1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.point_step * points.shape[0]
    message.is_dense = int(np.isfinite(points).all())
    message.data = structured.tobytes()
    return message


def hex_to_RGB(hex_color: str) -> list[int]:
    return [int(hex_color[i : i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(rgb: Sequence[int]) -> str:
    rgb = [int(value) for value in rgb]
    return "#" + "".join([f"0{value:x}" if value < 16 else f"{value:x}" for value in rgb])


def color_dict(gradient: Sequence[Sequence[int]]) -> dict[str, list]:
    return {
        "hex": [RGB_to_hex(rgb) for rgb in gradient],
        "r": [rgb[0] for rgb in gradient],
        "g": [rgb[1] for rgb in gradient],
        "b": [rgb[2] for rgb in gradient],
    }


def linear_gradient(start_hex: str, finish_hex: str = "#FFFFFF", n: int = 10) -> dict[str, list]:
    start_rgb = hex_to_RGB(start_hex)
    finish_rgb = hex_to_RGB(finish_hex)
    rgb_list = [start_rgb]

    for step in range(1, n):
        current = [
            int(start_rgb[channel] + (float(step) / (n - 1)) * (finish_rgb[channel] - start_rgb[channel]))
            for channel in range(3)
        ]
        rgb_list.append(current)

    return color_dict(rgb_list)


def rand_hex_color(num: int = 1) -> str | list[str]:
    colors = [RGB_to_hex([value * 255 for value in rnd.rand(3)]) for _ in range(num)]
    return colors[0] if num == 1 else colors


def polylinear_gradient(colors: Sequence[str], n: int) -> dict[str, list]:
    n_out = int(float(n) / (len(colors) - 1))
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for index in range(1, len(colors) - 1):
            next_gradient = linear_gradient(colors[index], colors[index + 1], n_out)
            for key in ("hex", "r", "g", "b"):
                gradient_dict[key] += next_gradient[key][1:]

    return gradient_dict


def color_map(data: np.ndarray, colors: Sequence[str], nLevels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradient_dict = polylinear_gradient(colors, nLevels)
    steps = len(gradient_dict["hex"])
    data_min = data.min()
    data_max = data.max()
    step = (data_max - data_min) / (steps - 1)

    step_indices = ((data - data_min) / step).astype(np.int32)
    r_array = np.array(gradient_dict["r"])
    g_array = np.array(gradient_dict["g"])
    b_array = np.array(gradient_dict["b"])
    return r_array[step_indices], g_array[step_indices], b_array[step_indices]


if __name__ == "__main__":
    palette = ["#2980b9", "#27ae60", "#f39c12", "#c0392b"]
    sample_data = np.linspace(0, 99, 100)
    color_map(sample_data, palette, 20)
