"""Compatibility wrappers around the shared octree implementation for scene modeling."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from oct_Tree import oct_tree_simple as core
import tools as scene_tools


env_size = core.DEFAULT_ENV_SIZE
voxel_size = core.DEFAULT_VOXEL_SIZE
max_depth = core.DEFAULT_MAX_DEPTH
DEFAULT_SCENE_COUNT = 1000


class OctreeNode(core.OctreeNode):
    """Scene-modeling octree node with unique-voxel occupancy scoring."""

    def compute_p(self) -> float:
        voxel_indices = {
            tuple(np.floor(np.asarray(point, dtype=float) / self.voxel_size).astype(int))
            for point in self.points
            if np.all(np.abs(point) <= 0.5 * self.base_size)
        }
        return (len(voxel_indices) * self.voxel_size * self.voxel_size) / (self.size * self.size)


def compute_index(parent: Sequence[float], son: Sequence[float]) -> int:
    """Compatibility wrapper around the shared octant index helper."""
    return core.compute_index(parent, son)


def compute_index_array(
    point: Sequence[float],
    total_size: float = env_size,
    min_size: float = voxel_size,
    origin: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compatibility wrapper around the shared octant path helper."""
    return core.compute_index_array(point, total_size=total_size, min_size=min_size, origin=origin)


def insert_node(root: OctreeNode, point: Sequence[float], maxdepth: int = max_depth - 1) -> None:
    """Insert a point into the scene-modeling octree using the shared path logic."""
    point_array = np.asarray(point, dtype=float)
    path = compute_index_array(point_array, total_size=root.base_size, min_size=root.voxel_size)
    current = root
    current.points.append(point_array)

    for depth_index, child_index in enumerate(path[:maxdepth], start=2):
        if current.children[child_index] is None:
            current.children[child_index] = OctreeNode(
                index=child_index,
                depth=depth_index,
                parent=current,
                base_size=root.base_size,
                voxel_size=root.voxel_size,
                density_weights=root.density_weights,
            )
        current = current.children[child_index]
        current.points.append(point_array)


def compute_layer_similarity(
    empirical_node: OctreeNode,
    current_node: OctreeNode,
    weights=core.DEFAULT_SIMILARITY_WEIGHTS,
    max_depth_value: int = max_depth,
    similarity_threshold: float = 0.0,
) -> float:
    """Recursively compute scene similarity with the historical scene-modeling rules."""
    if current_node.depth == max_depth_value:
        return similarity_threshold

    similarity_sum = 0.0
    weight_1, weight_2 = weights
    for child_index in range(8):
        empirical_child = empirical_node.children[child_index]
        current_child = current_node.children[child_index]
        if empirical_child is not None and current_child is not None:
            similarity_sum += compute_layer_similarity(
                empirical_child,
                current_child,
                weights=weights,
                max_depth_value=max_depth_value,
                similarity_threshold=similarity_threshold,
            )
            continue
        if empirical_child is None and current_child is None:
            similarity_sum += 1.0
            continue
        if current_node.depth == max_depth_value - 1:
            similarity_sum += similarity_threshold
            continue

        occupancy_current = current_child.compute_p() if current_child is not None else 0.0
        occupancy_empirical = empirical_child.compute_p() if empirical_child is not None else 0.0
        density = current_child.compute_rou() if current_child is not None else empirical_child.compute_rou()
        similarity_sum += weight_1 * abs(occupancy_current - occupancy_empirical) + weight_2 * density

    return similarity_sum / 8.0


def compute_similarity(root1: OctreeNode, root2: OctreeNode) -> float:
    """Compute similarity between two scene-modeling octrees."""
    return compute_layer_similarity(root1, root2)


def generate_octree_from_txt(path: str) -> OctreeNode:
    """Load one scene text file and build the scene-modeling octree."""
    points = scene_tools.read_data_from_txt(path)
    filtered_points = scene_tools.filter_scene_data(
        points=points,
        bound_low=core.DEFAULT_SCENE_LOW,
        bound_up=core.DEFAULT_SCENE_UP,
        center=core.DEFAULT_SCENE_CENTER,
    )
    root = OctreeNode(index=0, depth=1, parent=None)
    for point in filtered_points:
        insert_node(root, point)
    return root


def get_best_experience(base_size: int, current_scene: OctreeNode) -> int:
    """Return the most similar historical scene index for the current octree."""
    best_similarity = -np.inf
    best_index = -1
    for index in range(base_size):
        candidate_root = generate_octree_from_txt("scene_data/data_of_scene/data_of_scene{0}.txt".format(index))
        similarity = compute_similarity(current_scene, candidate_root)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = index
    return best_index


__all__ = [
    "OctreeNode",
    "compute_index",
    "compute_index_array",
    "compute_layer_similarity",
    "compute_similarity",
    "generate_octree_from_txt",
    "get_best_experience",
    "insert_node",
]
