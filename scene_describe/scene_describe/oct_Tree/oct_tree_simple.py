"""Octree utilities for scene similarity and potential-field generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .scene_data import tools as scene_tools
except ImportError:  # pragma: no cover - script execution fallback
    from oct_Tree.scene_data import tools as scene_tools


DEFAULT_ENV_SIZE = 0.16
DEFAULT_VOXEL_SIZE = 0.02
DEFAULT_MAX_DEPTH = 4
DEFAULT_SCENE_LOW = np.array([0.46, 0.0, 0.48], dtype=float)
DEFAULT_SCENE_UP = np.array([0.7, 0.2, 0.68], dtype=float)
DEFAULT_SCENE_CENTER = np.array([0.58, 0.08, 0.57], dtype=float)
DEFAULT_DENSITY_WEIGHTS = (0.6, 0.4)
DEFAULT_SIMILARITY_WEIGHTS = (0.5, 0.5)
DEFAULT_FORCE_DISTANCE = 0.08
DEFAULT_FORCE_EXPONENT = 2.0

OCTANT_DIRECTIONS = {
    0: np.array([1.0, 1.0, 1.0]),
    1: np.array([-1.0, 1.0, 1.0]),
    2: np.array([-1.0, -1.0, 1.0]),
    3: np.array([1.0, -1.0, 1.0]),
    4: np.array([1.0, 1.0, -1.0]),
    5: np.array([-1.0, 1.0, -1.0]),
    6: np.array([-1.0, -1.0, -1.0]),
    7: np.array([1.0, -1.0, -1.0]),
}
DEPTH_WEIGHT_DENOMINATOR = sum(math.exp(1.0 + 1.0 / depth) for depth in range(2, DEFAULT_MAX_DEPTH + 1))


@dataclass
class ArtiPotentialPoint:
    """Potential-field point generated from octree differences."""

    attribute: int
    depth: int
    index_array: Sequence[int]
    fi: float = DEFAULT_FORCE_DISTANCE
    n: float = DEFAULT_FORCE_EXPONENT
    center: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.attribute not in (0, 1):
            raise ValueError("Potential point attribute must be 0 (repulsive) or 1 (attractive).")
        self.center = compute_center_from_index(self.index_array)

    def get_force(self, point: Sequence[float]) -> np.ndarray:
        """Compute the virtual force that this potential point applies to a query point."""
        query_point = np.asarray(point, dtype=float)
        direction = self.center - query_point
        distance = np.linalg.norm(direction)
        if distance == 0.0 or distance > self.fi:
            return np.zeros(3, dtype=float)

        direction /= distance
        depth_weight = math.exp(1.0 + 1.0 / self.depth) / DEPTH_WEIGHT_DENOMINATOR
        if self.attribute == 1:
            magnitude = depth_weight * math.pow(distance, 1.0 / self.n)
            return magnitude * direction

        magnitude = depth_weight / math.pow(distance, self.n)
        return -magnitude * direction


@dataclass
class OctreeNode:
    """One node in the local scene octree."""

    index: int = -1
    depth: int = 1
    parent: Optional["OctreeNode"] = None
    base_size: float = DEFAULT_ENV_SIZE
    voxel_size: float = DEFAULT_VOXEL_SIZE
    density_weights: Tuple[float, float] = DEFAULT_DENSITY_WEIGHTS
    children: List[Optional["OctreeNode"]] = field(default_factory=lambda: [None] * 8)
    points: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.size = self.base_size / (2 ** (self.depth - 1))

    def __str__(self) -> str:
        return "Node(index={0}, depth={1})".format(self.index, self.depth)

    def compute_p(self) -> float:
        """Compute the occupancy ratio of this node using the stored point count."""
        return (len(self.points) * self.voxel_size * self.voxel_size) / (self.size * self.size)

    def compute_rou(self) -> float:
        """Compute the recursive density term used by the original similarity metric."""
        valid_children = [child for child in self.children if child is not None]
        count = len(valid_children)
        if self.depth == DEFAULT_MAX_DEPTH - 1:
            return count / 8.0
        if not valid_children:
            return 0.0

        child_density = sum(child.compute_rou() for child in valid_children)
        weight_1, weight_2 = self.density_weights
        return weight_1 * (count / 8.0) + weight_2 * (child_density / count)

    def compute_occupancy_ratio(self) -> float:
        """English alias for ``compute_p``."""
        return self.compute_p()

    def compute_density_ratio(self) -> float:
        """English alias for ``compute_rou``."""
        return self.compute_rou()

    def show_tree(self) -> None:
        """Print this node and all descendants."""
        print(self)
        for child in self.children:
            if child is not None:
                child.show_tree()


def compute_index(parent: Sequence[float], son: Sequence[float]) -> int:
    """Return the octant index of ``son`` relative to ``parent``."""
    relative = np.asarray(son, dtype=float) - np.asarray(parent, dtype=float)
    x_positive, y_positive, z_positive = (relative > 0.0).astype(int)
    return {
        (1, 1, 1): 0,
        (0, 1, 1): 1,
        (0, 0, 1): 2,
        (1, 0, 1): 3,
        (1, 1, 0): 4,
        (0, 1, 0): 5,
        (0, 0, 0): 6,
        (1, 0, 0): 7,
    }[(x_positive, y_positive, z_positive)]


comput_index = compute_index


def compute_index_array(
    point: Sequence[float],
    total_size: float = DEFAULT_ENV_SIZE,
    min_size: float = DEFAULT_VOXEL_SIZE,
    origin: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute the hierarchical octant path for one point."""
    point_array = np.asarray(point, dtype=float)
    center = np.zeros(3, dtype=float) if origin is None else np.asarray(origin, dtype=float)
    layer_count = int(np.log2(total_size / min_size))
    indices = []

    for layer in range(layer_count):
        index = compute_index(center, point_array)
        indices.append(index)
        center = center + OCTANT_DIRECTIONS[index] * total_size / (2 ** (layer + 2))
    return np.asarray(indices, dtype=int)


def insert_node(root: OctreeNode, point: Sequence[float], max_depth: int = DEFAULT_MAX_DEPTH - 1) -> None:
    """Insert one point into the octree up to ``max_depth`` levels below the root."""
    if root is None:
        raise ValueError("Root node cannot be None.")

    point_array = np.asarray(point, dtype=float)
    path = compute_index_array(point_array, total_size=root.base_size, min_size=root.voxel_size)
    current = root
    current.points.append(point_array)

    for depth_index, child_index in enumerate(path[:max_depth], start=2):
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
    node_empir: OctreeNode,
    node_current: OctreeNode,
    weights: Tuple[float, float] = DEFAULT_SIMILARITY_WEIGHTS,
) -> float:
    """Recursively compute the original octree similarity metric for one layer."""
    depth = node_current.depth
    if depth == DEFAULT_MAX_DEPTH:
        return 1.0

    similarity = 0.0
    weight_1, weight_2 = weights
    for child_index in range(8):
        empirical_child = node_empir.children[child_index]
        current_child = node_current.children[child_index]
        if empirical_child is not None and current_child is not None:
            similarity += compute_layer_similarity(empirical_child, current_child, weights=weights)
            continue
        if empirical_child is None and current_child is None:
            similarity += 1.0
            continue

        if depth == DEFAULT_MAX_DEPTH - 1:
            similarity += 0.0
            continue

        occupancy_current = current_child.compute_p() if current_child is not None else 0.0
        occupancy_empirical = empirical_child.compute_p() if empirical_child is not None else 0.0
        density = current_child.compute_rou() if current_child is not None else empirical_child.compute_rou()
        similarity += weight_1 * abs(occupancy_current - occupancy_empirical) + weight_2 * density

    return similarity / 8.0


def compute_layer_sim(node_empir: OctreeNode, node_current: OctreeNode) -> float:
    """Compatibility wrapper for the original function name."""
    return compute_layer_similarity(node_empir, node_current)


def compute_similarity(root1: OctreeNode, root2: OctreeNode) -> float:
    """Compute the similarity score between two octree roots."""
    return compute_layer_similarity(root1, root2)


def compute_sim(root1: OctreeNode, root2: OctreeNode) -> float:
    """Compatibility wrapper for the original function name."""
    return compute_similarity(root1, root2)


def _build_index_array(node: OctreeNode, child_index: int) -> List[int]:
    indices = [child_index]
    current = node
    while current.parent is not None:
        if current.parent.depth != 1:
            indices.append(current.parent.index)
        current = current.parent
    indices.reverse()
    return indices


def compute_potential(root_history: OctreeNode, root_current: OctreeNode) -> List[ArtiPotentialPoint]:
    """Generate attractive and repulsive potential points by comparing two octrees."""
    potential_points = []
    if root_current.depth == DEFAULT_MAX_DEPTH:
        return potential_points

    for child_index in range(8):
        history_child = root_history.children[child_index]
        current_child = root_current.children[child_index]

        if history_child is not None and current_child is None:
            potential_points.append(
                ArtiPotentialPoint(
                    attribute=1,
                    depth=history_child.depth,
                    index_array=_build_index_array(root_history, child_index),
                )
            )
        elif current_child is not None and history_child is None:
            potential_points.append(
                ArtiPotentialPoint(
                    attribute=0,
                    depth=current_child.depth,
                    index_array=_build_index_array(root_current, child_index),
                )
            )
        elif history_child is not None and current_child is not None:
            potential_points.extend(compute_potential(history_child, current_child))

    return potential_points


def compute_center_from_index(index_array: Sequence[int]) -> np.ndarray:
    """Compute the center position of an octree node from its index path."""
    distance = DEFAULT_ENV_SIZE / 4.0
    center = np.zeros(3, dtype=float)
    for index in index_array:
        center += distance * OCTANT_DIRECTIONS[index]
        distance /= 2.0
    return center


def compute_total_force(potential_points: Iterable[ArtiPotentialPoint], point: Sequence[float]) -> np.ndarray:
    """Sum all potential-field forces applied to one query point."""
    total_force = np.zeros(3, dtype=float)
    for potential_point in potential_points:
        total_force += potential_point.get_force(point)
    return total_force


def generate_octree_from_points(points: np.ndarray) -> OctreeNode:
    """Create an octree from already-centered scene points."""
    root = OctreeNode(index=0, depth=1, parent=None)
    for point in np.asarray(points, dtype=float):
        insert_node(root, point)
    return root


def generate_octree_from_txt(
    path: str,
    bound_low: Sequence[float] = DEFAULT_SCENE_LOW,
    bound_up: Sequence[float] = DEFAULT_SCENE_UP,
    center: Sequence[float] = DEFAULT_SCENE_CENTER,
) -> OctreeNode:
    """Load a legacy text scene file, crop it, center it, and build the octree."""
    points = scene_tools.read_scene_data_in_txt(path)
    filtered_points = scene_tools.filter_scene_data(points=points, bound_low=bound_low, bound_up=bound_up, center=center)
    return generate_octree_from_points(filtered_points)


__all__ = [
    "ArtiPotentialPoint",
    "OctreeNode",
    "compute_center_from_index",
    "compute_index",
    "compute_index_array",
    "compute_layer_sim",
    "compute_layer_similarity",
    "compute_potential",
    "compute_sim",
    "compute_similarity",
    "compute_total_force",
    "comput_index",
    "generate_octree_from_points",
    "generate_octree_from_txt",
    "insert_node",
]
