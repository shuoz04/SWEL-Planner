"""Scene data readers and filters used by the octree experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TEXT_SECTION_MARKERS = {
    "下面是x坐标": 0,
    "下面是y坐标": 1,
    "下面是z坐标": 2,
}


def read_3d_data(path: str) -> np.ndarray:
    """Read 3D point data from the first sheet of an Excel file."""
    data_frame = pd.read_excel(path, sheet_name="Sheet1")
    return data_frame.to_numpy(dtype=float)


def filter_scene_data(
    points: np.ndarray,
    bound_low: Sequence[float],
    bound_up: Sequence[float],
    center: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Filter 3D points by axis-aligned bounds and optionally subtract a center offset."""
    point_array = np.asarray(points, dtype=float)
    lower = np.asarray(bound_low, dtype=float)
    upper = np.asarray(bound_up, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("Scene points must be an Nx3 array.")
    if np.any(upper < lower):
        raise ValueError("Each upper bound must be greater than or equal to the lower bound.")

    mask = np.all((point_array >= lower) & (point_array <= upper), axis=1)
    filtered = point_array[mask].copy()
    if center is not None:
        filtered -= np.asarray(center, dtype=float)
    return filtered


def read_scene_data_in_txt(file_path: str) -> np.ndarray:
    """Read the legacy text format that stores x, y, and z coordinates in separate blocks."""
    coordinate_buffers = [[], [], []]
    current_axis = None

    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in TEXT_SECTION_MARKERS:
                current_axis = TEXT_SECTION_MARKERS[stripped]
                continue
            if current_axis is None:
                raise ValueError(
                    "Line {0} in {1} appears before any coordinate section header.".format(
                        line_number,
                        file_path,
                    )
                )
            coordinate_buffers[current_axis].append(float(stripped))

    x_count, y_count, z_count = map(len, coordinate_buffers)
    if x_count != y_count or y_count != z_count:
        raise ValueError(
            "Coordinate counts do not match in {0}: x={1}, y={2}, z={3}.".format(
                file_path,
                x_count,
                y_count,
                z_count,
            )
        )
    return np.column_stack(coordinate_buffers).astype(float)


__all__ = [
    "filter_scene_data",
    "read_3d_data",
    "read_scene_data_in_txt",
]
