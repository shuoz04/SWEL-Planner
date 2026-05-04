"""Potential-field based GMM mean adjustment for scene adaptation experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from GMM.GMM import WeightedGMM
from oct_Tree.oct_tree_simple import ArtiPotentialPoint, compute_potential, compute_total_force
from oct_tree import OctreeNode, generate_octree_from_txt, insert_node
import tools as scene_tools


DEFAULT_SCENE_CENTER = np.array([0.397, 0.0833, 0.583], dtype=float)
DEFAULT_BASELINE_EXCEL = "scene_data/baseline_scene.xlsx"
DEFAULT_SCENE_INDEX_BY_RATIO = {
    20: 31,
    30: 239,
    40: 32,
    50: 964,
    60: 40,
    70: 900,
    80: 64,
    90: 639,
}


def build_current_scene_from_excel(excel_path: str = DEFAULT_BASELINE_EXCEL) -> OctreeNode:
    """Build the reference scene octree from the baseline Excel file."""
    points = scene_tools.read_3d_data(excel_path)
    filtered_points = scene_tools.filter_scene_data(
        points=points,
        bound_low=(0.46, 0.0, 0.48),
        bound_up=(0.7, 0.2, 0.68),
        center=(0.58, 0.08, 0.57),
    )
    root = OctreeNode(index=0, depth=1, parent=None)
    for point in filtered_points:
        insert_node(root, point)
    return root


def _normalize_force(force: Sequence[float]) -> np.ndarray:
    force_array = np.asarray(force, dtype=float)
    norm = np.linalg.norm(force_array)
    if norm == 0.0:
        return np.zeros_like(force_array)
    return force_array / norm


def get_gmm_finetuning(
    pkl_path: str,
    scene_num: int,
    step_size: float,
    current_scene: OctreeNode,
    new_pkl_path: str,
    center: Sequence[float] = DEFAULT_SCENE_CENTER,
) -> WeightedGMM:
    """Adjust GMM means using the potential field between a history scene and the current scene."""
    center_array = np.asarray(center, dtype=float)
    with open(pkl_path, "rb") as handle:
        gmm_pos: WeightedGMM = pickle.load(handle)

    history_path = "scene_data/data_of_scene/data_of_scene{0}.txt".format(scene_num)
    history_root = generate_octree_from_txt(history_path)
    potential_points = compute_potential(root_history=history_root, root_current=current_scene)

    updated_means = []
    for mean in np.asarray(gmm_pos.mu, dtype=float):
        force = compute_total_force(potential_points, np.asarray(mean, dtype=float) - center_array)
        updated_means.append(mean + step_size * _normalize_force(force))
    gmm_pos.mu = np.asarray(updated_means, dtype=float)

    output_path = Path(new_pkl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(gmm_pos, handle)
    return gmm_pos


def run_finetuning_batch(
    ratios: Iterable[int],
    step_sizes_cm: Iterable[int],
    current_scene: OctreeNode,
    scene_index_by_ratio: Dict[int, int] = DEFAULT_SCENE_INDEX_BY_RATIO,
) -> None:
    """Run the historical finetuning batch for a set of experience ratios and step sizes."""
    for step_size_cm in step_sizes_cm:
        step_size = step_size_cm * 0.01
        for ratio in ratios:
            pkl_path = "../GMM/data/GMM_model/pos_model/gmm_{0}%_pos.pkl".format(ratio)
            output_path = (
                "../GMM/data/GMM_model/pos_model/pos_finetuning/step_size{0}cm/"
                "gmm_{1}%_pos_finetuning.pkl"
            ).format(step_size_cm, ratio)
            get_gmm_finetuning(
                pkl_path=pkl_path,
                scene_num=scene_index_by_ratio[ratio],
                step_size=step_size,
                current_scene=current_scene,
                new_pkl_path=output_path,
            )


__all__ = [
    "ArtiPotentialPoint",
    "OctreeNode",
    "build_current_scene_from_excel",
    "compute_potential",
    "compute_total_force",
    "generate_octree_from_txt",
    "get_gmm_finetuning",
    "run_finetuning_batch",
]


if __name__ == "__main__":
    baseline_scene = build_current_scene_from_excel()
    run_finetuning_batch(ratios=[30], step_sizes_cm=[3], current_scene=baseline_scene)
