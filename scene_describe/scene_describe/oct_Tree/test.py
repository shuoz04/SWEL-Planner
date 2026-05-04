"""Batch evaluation helpers for the legacy octree scene similarity experiments."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from . import oct_tree_simple as octree
    from .scene_data import tools as scene_tools
except ImportError:  # pragma: no cover - script execution fallback
    import oct_tree_simple as octree
    from scene_data import tools as scene_tools


DEFAULT_BASELINE_EXCEL = "./scene_data/data_of_scene1.xlsx"
DEFAULT_SCENE_TEMPLATE = "scene_data/data_of_scene/data_of_scene{0}.txt"


def generate_octree_from_txt(path: str) -> octree.OctreeNode:
    """Compatibility wrapper that builds an octree from one legacy scene text file."""
    return octree.generate_octree_from_txt(path)


def build_baseline_octree(excel_path: str = DEFAULT_BASELINE_EXCEL) -> octree.OctreeNode:
    """Build the empirical reference octree from the baseline Excel scene."""
    points = scene_tools.read_3d_data(excel_path)
    filtered = scene_tools.filter_scene_data(
        points=points,
        bound_low=octree.DEFAULT_SCENE_LOW,
        bound_up=octree.DEFAULT_SCENE_UP,
        center=octree.DEFAULT_SCENE_CENTER,
    )
    return octree.generate_octree_from_points(filtered)


def evaluate_scene_similarity(root_empirical: octree.OctreeNode, scene_count: int = 1000) -> np.ndarray:
    """Compute similarity scores for a batch of legacy scene files."""
    roots = [generate_octree_from_txt(DEFAULT_SCENE_TEMPLATE.format(index)) for index in range(scene_count)]
    return np.asarray([octree.compute_similarity(root_empirical, root) for root in roots], dtype=float)


def summarize_similarity_bins(similarities: np.ndarray) -> Dict[str, float]:
    """Return coarse similarity ratios for the historic strong and middle bins."""
    total = max(len(similarities), 1)
    strong_ratio = float(np.sum(similarities > 0.7)) / total
    middle_ratio = float(np.sum((similarities > 0.5) & (similarities <= 0.7))) / total
    return {
        "strong_ratio": strong_ratio,
        "middle_ratio": middle_ratio,
        "max_similarity": float(np.max(similarities)) if len(similarities) else 0.0,
    }


def main() -> None:
    """Run the legacy octree batch similarity evaluation."""
    baseline_root = build_baseline_octree()
    similarities = evaluate_scene_similarity(baseline_root)
    summary = summarize_similarity_bins(similarities)
    print("strong ratio:", summary["strong_ratio"])
    print("middle ratio:", summary["middle_ratio"])
    print("max similarity:", summary["max_similarity"])


if __name__ == "__main__":
    main()
