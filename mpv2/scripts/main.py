#!/usr/bin/env python3
"""Entry points for the MPV2 planning demos."""

from __future__ import annotations

import sys
import time
from typing import Callable, Sequence

import numpy as np
import rospy

from planner.moveit_spaces import (
    MoveItCartesianDecomposition,
    MoveItGroupConfig,
    MoveItJointSpace,
    PlanarDecomposition,
)
from planner.stp import STP, State


PLANAR_GROUP = MoveItGroupConfig(
    group_name="arm",
    joint_names=("joint_1", "joint_2", "joint_3"),
)

JAKA_GROUP = MoveItGroupConfig(
    group_name="arm",
    joint_names=tuple(f"right_joint_{i}" for i in range(1, 7)),
    tool_link_name="right_gripper_base_link",
    frame_id="right_base_link",
    seed_bounds=(
        (-4.963716393, -1.171988593, -2.412917691, -1.171988593, -4.963716393, -4.963716393),
        (4.963716393, 3.653846789, 2.412917691, 3.653846789, 4.963716393, 4.963716393),
    ),
    free_volume_overrides={(0, 0, 0): 0.2},
)


class PlanarSpace(MoveItJointSpace):
    def __init__(self, lb: tuple[float, ...], ub: tuple[float, ...]):
        super().__init__(lb, ub, PLANAR_GROUP)


class PlanarDecomp(PlanarDecomposition):
    def __init__(self, lb: tuple[float, ...], ub: tuple[float, ...], slices: tuple[int, ...]):
        super().__init__(lb, ub, slices, free_volume_overrides={(0, 0): 0.1})


class JakaSpace(MoveItJointSpace):
    def __init__(self, lb: tuple[float, ...], ub: tuple[float, ...]):
        super().__init__(lb, ub, JAKA_GROUP)


class JakaDecomp(MoveItCartesianDecomposition):
    def __init__(self, lb: tuple[float, ...], ub: tuple[float, ...], slices: tuple[int, ...]):
        super().__init__(lb, ub, slices, config=JAKA_GROUP)


def _measure_solve(planner: STP, start: State, goal: State) -> None:
    started_at = time.time()
    result = planner.solve(start, goal)
    elapsed_ms = (time.time() - started_at) * 1000.0
    print(f"res: {result}")
    print(f"cost: {elapsed_ms:.2f} ms")


def _build_planar_planner() -> STP:
    space = PlanarSpace((-np.pi, -np.pi, -np.pi), (np.pi, np.pi, np.pi))
    decomp = PlanarDecomp((-1.1, -1.1, -np.pi), (1.1, 1.1, np.pi), (9, 7))
    return STP(space, decomp)


def _build_jaka_planner(
    workspace_lb: tuple[float, ...] = (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
    workspace_ub: tuple[float, ...] = (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
    slices: tuple[int, ...] = (12, 12, 12),
) -> STP:
    lower_bounds, upper_bounds = JAKA_GROUP.seed_bounds
    space = JakaSpace(lower_bounds, upper_bounds)
    decomp = JakaDecomp(workspace_lb, workspace_ub, slices)
    return STP(space, decomp)


def planar() -> None:
    planner = _build_planar_planner()
    start = State(*np.deg2rad([0, 0, 30]))
    goal = State(*np.deg2rad([120, -60, 30]))
    _measure_solve(planner, start, goal)


def jaka() -> None:
    planner = _build_jaka_planner()
    start = State(*np.deg2rad([-30, 60, -45, -17, -115, 0]))
    goal = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))
    _measure_solve(planner, start, goal)


def plot() -> None:
    planner = _build_jaka_planner()
    start = State(*np.deg2rad([-30, 60, -45, -17, -115, 0]))
    goal = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))
    planner.plot2(start, goal)


def plot2() -> None:
    planner = _build_jaka_planner(
        workspace_lb=(-0.3, -0.3, 0.0, -np.pi, -np.pi, -np.pi),
        workspace_ub=(0.9, 0.66, 0.96, np.pi, np.pi, np.pi),
        slices=(10, 8, 8),
    )
    planner.plot3()


def main(argv: Sequence[str] | None = None) -> None:
    commands: dict[str, Callable[[], None]] = {
        "planar": planar,
        "jaka": jaka,
        "plot": plot,
        "plot2": plot2,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    command_name = args[0] if args else "plot"

    if command_name not in commands:
        valid_commands = ", ".join(sorted(commands))
        raise SystemExit(f"Unknown command '{command_name}'. Available commands: {valid_commands}")

    if not rospy.core.is_initialized():
        rospy.init_node("main")
    commands[command_name]()


if __name__ == "__main__":
    main()
