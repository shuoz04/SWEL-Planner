#!/usr/bin/env python3
"""Legacy planning demos built on top of the shared MPV2 planner stack.

This module keeps the historic ``hiey_test.py`` entry points available while
moving the implementation onto the shared planner helpers in
``planner.moveit_spaces`` and ``planner.stp``. The goal is to preserve the demo
surface area without carrying duplicated math, duplicated MoveIt wiring, or
script-only structure.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rospy

from planner.gmm import (
    WeightedGMM as SharedWeightedGMM,
    exp_map as shared_exp_map,
    log_map as shared_log_map,
    quaternion_multiply as shared_quaternion_multiply,
    weighted_quaternion_mean as shared_weighted_quaternion_mean,
)
from planner.moveit_spaces import (
    MoveItCartesianDecomposition,
    MoveItGroupConfig,
    MoveItJointSpace,
    PlanarDecomposition,
)
from planner.stp import Cell, Decomposition, Magic, STP, Space, State, wrap_to_pi


LOGGER = logging.getLogger(__name__)
Bounds = Tuple[Tuple[float, ...], Tuple[float, ...]]
WorkspaceSpec = Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[int, ...]]

PLANAR_BOUNDS = (
    (-np.pi, -np.pi, -np.pi),
    (np.pi, np.pi, np.pi),
)
PLANAR_WORKSPACE = (
    (-1.1, -1.1, -np.pi),
    (1.1, 1.1, np.pi),
    (9, 7),
)

LEGACY_JAKA_6_BOUNDS = (
    (-4.963716393, -1.171988593, -2.412917691, -1.171988593, -4.963716393, -4.963716393),
    (4.963716393, 3.653846789, 2.412917691, 3.653846789, 4.963716393, 4.963716393),
)
RM75_7_BOUNDS = (
    (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
    (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi),
)

DEFAULT_WORKSPACE = (
    (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
    (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
    (12, 12, 12),
)
PLOT_WORKSPACE = (
    (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
    (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
    (12, 12, 12),
)
PLOT2_WORKSPACE = (
    (-0.3, -0.3, 0.0, -np.pi, -np.pi, -np.pi),
    (0.9, 0.66, 0.96, np.pi, np.pi, np.pi),
    (10, 8, 8),
)
TASK_WORKSPACE = (
    (0.397 - 0.04, 0.0833 - 0.04, 0.583 - 0.04, -np.pi, -np.pi, -np.pi),
    (0.397 + 0.04, 0.0833 + 0.04, 0.583 + 0.04, np.pi, np.pi, np.pi),
    (1, 1, 1),
)

DEFAULT_FREE_VOLUME_OVERRIDES = {(0, 0, 0): 0.2}
PLANAR_FREE_VOLUME_OVERRIDES = {(0, 0): 0.1}

DEFAULT_TOOL_LINK_NAME = "tool_link"
DEFAULT_FRAME_ID = "base_link"
DEFAULT_GROUP_NAME = "arm"
DEFAULT_MAIN_COMMAND = "task"

DEFAULT_POTENTIAL_POINTS = np.array(
    [

    ],
    dtype=float,
)

PLANAR_START = State(*np.deg2rad([0, 0, 30]))
PLANAR_GOAL = State(*np.deg2rad([120, -60, 30]))
JAKA_START = State(*np.deg2rad([-30, 60, -45, -17, -115, 0]))
JAKA_GOAL = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))
TASK_START = State(*np.deg2rad([0, 0, 0, 30, 0, 60, 100]))
TASK_GOAL = State(
    *np.deg2rad([32.939617, 27.272095, -8.98766, 43.592518, -82.58138, 60.732075, 145.08644])
)
DEFAULT_POTENTIAL_SEED = State(*np.deg2rad([0, 120, -90, 150, 90, 0]))

weighted_quaternion_mean = shared_weighted_quaternion_mean
log_map = shared_log_map
exp_map = shared_exp_map
quaternion_multiply = shared_quaternion_multiply
WeightedGMM = SharedWeightedGMM

PLANAR_GROUP = MoveItGroupConfig(
    group_name=DEFAULT_GROUP_NAME,
    joint_names=("joint_1", "joint_2", "joint_3"),
)


def _joint_names_for_dimension(joint_count: int) -> Tuple[str, ...]:
    if joint_count == 3:
        return PLANAR_GROUP.joint_names
    if joint_count == 6:
        return tuple("right_joint_{0}".format(index) for index in range(1, 7))
    if joint_count == 7:
        return tuple("joint{0}".format(index) for index in range(1, 8))
    return tuple("joint{0}".format(index) for index in range(1, joint_count + 1))


def _build_moveit_group_config(
    seed_bounds: Bounds,
    joint_names: Optional[Tuple[str, ...]] = None,
    tool_link_name: str = DEFAULT_TOOL_LINK_NAME,
    frame_id: str = DEFAULT_FRAME_ID,
    free_volume_overrides: Optional[Dict[Tuple[int, ...], float]] = None,
    orientation_copy_probability: float = 0.95,
) -> MoveItGroupConfig:
    resolved_joint_names = joint_names or _joint_names_for_dimension(len(seed_bounds[0]))
    return MoveItGroupConfig(
        group_name=DEFAULT_GROUP_NAME,
        joint_names=resolved_joint_names,
        tool_link_name=tool_link_name,
        frame_id=frame_id,
        seed_bounds=seed_bounds,
        orientation_copy_probability=orientation_copy_probability,
        free_volume_overrides=dict(free_volume_overrides or DEFAULT_FREE_VOLUME_OVERRIDES),
    )


class PlanarSpace(MoveItJointSpace):
    """3-link planar arm validity checks backed by MoveIt."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...]) -> None:
        super().__init__(lb, ub, PLANAR_GROUP)


class PlanarDecomp(PlanarDecomposition):
    """Planar analytical decomposition kept for legacy demos."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...], slices: Tuple[int, ...]) -> None:
        super().__init__(lb, ub, slices, free_volume_overrides=PLANAR_FREE_VOLUME_OVERRIDES)


class JakaSpace(MoveItJointSpace):
    """MoveIt-backed joint space that infers joint naming from the seed bounds."""

    def __init__(
        self,
        lb: Tuple[float, ...],
        ub: Tuple[float, ...],
        joint_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self._config = _build_moveit_group_config((lb, ub), joint_names=joint_names)
        super().__init__(lb, ub, self._config)


class JakaDecomp(MoveItCartesianDecomposition):
    """MoveIt workspace decomposition for the legacy Jaka/RM75 experiments."""

    def __init__(
        self,
        lb: Tuple[float, ...],
        ub: Tuple[float, ...],
        slices: Tuple[int, ...],
        seed_bounds: Bounds = LEGACY_JAKA_6_BOUNDS,
        joint_names: Optional[Tuple[str, ...]] = None,
        tool_link_name: str = DEFAULT_TOOL_LINK_NAME,
        frame_id: str = DEFAULT_FRAME_ID,
        free_volume_overrides: Optional[Dict[Tuple[int, ...], float]] = None,
        orientation_copy_probability: float = 0.95,
    ) -> None:
        self._config = _build_moveit_group_config(
            seed_bounds=seed_bounds,
            joint_names=joint_names,
            tool_link_name=tool_link_name,
            frame_id=frame_id,
            free_volume_overrides=free_volume_overrides,
            orientation_copy_probability=orientation_copy_probability,
        )
        super().__init__(lb, ub, slices, config=self._config)

    def ik_from_potential(self, point: Sequence[float], seed: State) -> Optional[State]:
        """Solve IK for a workspace point while reusing the seed orientation."""
        orientation = np.array(self.fk(seed).data_view[3:], dtype=float)
        workspace_values = np.hstack((np.asarray(point, dtype=float), orientation))
        workspace_state = State(*workspace_values)
        return self.moveit_ik(workspace_state, seed)


def _measure_solve(planner: STP, start: State, goal: State) -> None:
    started_at = time.time()
    result = planner.solve(start, goal)
    elapsed_ms = (time.time() - started_at) * 1000.0
    print("res: {0}".format(result))
    print("cost: {0:.2f} ms".format(elapsed_ms))


def build_planar_planner(interactive: bool = True) -> STP:
    """Create the historic planar demo planner."""
    space = PlanarSpace(*PLANAR_BOUNDS)
    decomp = PlanarDecomp(*PLANAR_WORKSPACE)
    return STP(space, decomp, interactive=interactive)


def build_jaka_planner(
    workspace: WorkspaceSpec = DEFAULT_WORKSPACE,
    seed_bounds: Bounds = LEGACY_JAKA_6_BOUNDS,
    joint_names: Optional[Tuple[str, ...]] = None,
    tool_link_name: str = DEFAULT_TOOL_LINK_NAME,
    frame_id: str = DEFAULT_FRAME_ID,
    free_volume_overrides: Optional[Dict[Tuple[int, ...], float]] = None,
    interactive: bool = True,
) -> STP:
    """Create a MoveIt-backed planner for the legacy Jaka/RM75 experiments."""
    space = JakaSpace(seed_bounds[0], seed_bounds[1], joint_names=joint_names)
    decomp = JakaDecomp(
        workspace[0],
        workspace[1],
        workspace[2],
        seed_bounds=seed_bounds,
        joint_names=joint_names,
        tool_link_name=tool_link_name,
        frame_id=frame_id,
        free_volume_overrides=free_volume_overrides,
    )
    return STP(space, decomp, interactive=interactive)


def planar() -> None:
    """Run the planar STP demo."""
    _measure_solve(build_planar_planner(), PLANAR_START, PLANAR_GOAL)


def jaka() -> None:
    """Run the 6-DOF MoveIt STP demo."""
    _measure_solve(build_jaka_planner(), JAKA_START, JAKA_GOAL)


def plot() -> None:
    """Replay the legacy RViz visualization path demo."""
    planner = build_jaka_planner(workspace=PLOT_WORKSPACE)
    planner.plot2(JAKA_START, JAKA_GOAL)


def plot2() -> None:
    """Publish the workspace decomposition used by the legacy visualization demo."""
    planner = build_jaka_planner(workspace=PLOT2_WORKSPACE)
    planner.plot3()


def task() -> None:
    """Run the data-collection experiment used by the historical scripts."""
    planner = build_jaka_planner(
        workspace=TASK_WORKSPACE,
        seed_bounds=RM75_7_BOUNDS,
        joint_names=_joint_names_for_dimension(7),
    )
    planner.collect_data(TASK_START, TASK_GOAL)


def compute_ik_results_from_potential(
    points: Sequence[Sequence[float]],
    seed: State = DEFAULT_POTENTIAL_SEED,
) -> List[State]:
    """Solve IK for a sequence of candidate potential-field points."""
    planner = build_jaka_planner()
    decomp = planner.decomp
    if not isinstance(decomp, JakaDecomp):
        raise TypeError("Potential-field IK requires a JakaDecomp instance.")

    solutions = []
    for point in points:
        state = decomp.ik_from_potential(point, seed)
        if state is not None:
            solutions.append(state)

    LOGGER.info("Resolved %d IK solutions from %d candidate points.", len(solutions), len(points))
    return solutions


def get_ik_results_from_potential() -> List[State]:
    """Compatibility wrapper for the legacy potential-field experiment."""
    solutions = compute_ik_results_from_potential(DEFAULT_POTENTIAL_POINTS)
    print("ik solutions: {0}".format(len(solutions)))
    return solutions


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point for the historical demo commands."""
    commands = {
        "planar": planar,
        "jaka": jaka,
        "plot": plot,
        "plot2": plot2,
        "task": task,
        "potential": get_ik_results_from_potential,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    command_name = args[0] if args else DEFAULT_MAIN_COMMAND

    if command_name not in commands:
        raise SystemExit(
            "Unknown command '{0}'. Available commands: {1}".format(
                command_name,
                ", ".join(sorted(commands)),
            )
        )

    if not rospy.core.is_initialized():
        rospy.init_node("main")
    commands[command_name]()


__all__ = [
    "Cell",
    "Decomposition",
    "JakaDecomp",
    "JakaSpace",
    "Magic",
    "PlanarDecomp",
    "PlanarSpace",
    "STP",
    "Space",
    "State",
    "WeightedGMM",
    "build_jaka_planner",
    "build_planar_planner",
    "compute_ik_results_from_potential",
    "exp_map",
    "get_ik_results_from_potential",
    "jaka",
    "log_map",
    "main",
    "planar",
    "plot",
    "plot2",
    "quaternion_multiply",
    "task",
    "weighted_quaternion_mean",
    "wrap_to_pi",
]


if __name__ == "__main__":
    main()
