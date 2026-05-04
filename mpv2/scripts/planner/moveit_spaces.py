#!/usr/bin/env python3
"""Reusable MoveIt-backed search spaces and decompositions for MPV2 demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import geometry_msgs.msg
import moveit_msgs.srv
import numpy as np
import rospy
import tf.transformations

from planner.stp import Cell, Decomposition, Magic, Space, State, wrap_to_pi


Bounds = Tuple[Tuple[float, ...], Tuple[float, ...]]


@dataclass(frozen=True)
class MoveItGroupConfig:
    """Runtime configuration for MoveIt-backed planner helpers."""

    group_name: str
    joint_names: Tuple[str, ...]
    tool_link_name: str | None = None
    frame_id: str | None = None
    seed_bounds: Bounds | None = None
    orientation_copy_probability: float = 0.95
    free_volume_overrides: Mapping[Tuple[int, ...], float] = field(default_factory=dict)
    check_state_service: str = "/check_state_validity"
    ik_service: str = "/compute_ik"
    fk_service: str = "/compute_fk"
    ik_timeout_sec: float = 1.0


class MoveItJointSpace(Space):
    """Joint-space validity checks backed by MoveIt state validity services."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...], config: MoveItGroupConfig):
        super().__init__(lb, ub)
        self._config = config
        self.check_validity_srv = rospy.ServiceProxy(config.check_state_service, moveit_msgs.srv.GetStateValidity)

    def check_validity(self, s: State) -> bool:
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = self._config.group_name
        req.robot_state.joint_state.name = list(self._config.joint_names)
        req.robot_state.joint_state.position = s.data_view.tolist()
        resp: moveit_msgs.srv.GetStateValidityResponse = self.check_validity_srv.call(req)
        return resp.valid


class PlanarDecomposition(Decomposition):
    """Workspace decomposition with analytical IK for a 3-link planar arm."""

    def __init__(
        self,
        lb: Tuple[float, ...],
        ub: Tuple[float, ...],
        slices: Tuple[int, ...],
        link_lengths: Tuple[float, float, float] = (0.5, 0.4, 0.1),
        free_volume_overrides: Mapping[Tuple[int, ...], float] | None = None,
        orientation_copy_probability: float = 0.9,
    ):
        self._link_lengths = link_lengths
        self._free_volume_overrides = dict(free_volume_overrides or {(0, 0): 0.1})
        self._orientation_copy_probability = orientation_copy_probability
        super().__init__(lb, ub, slices)

    def set_cell_free_vol(self) -> None:
        for rid, free_volume in self._free_volume_overrides.items():
            self._cells_dict[rid].free_vol = free_volume

    def _sample_in_cell(self, cell: Cell, seed: State | None) -> State | None:
        point = cell.ws.sample_uniform().data_view
        if seed is not None and np.random.random() < self._orientation_copy_probability:
            point[-1] = self.fk(seed).data_view[-1]

        states = self._analytical_ik(State(*point))
        if not states:
            return None

        if seed is None:
            return states[np.random.randint(0, len(states))]
        return min(states, key=lambda state: np.linalg.norm(state.data_view[:2] - seed.data_view[:2]))

    def _analytical_ik(self, workspace_state: State) -> list[State]:
        a, b, c = self._link_lengths
        x, y, theta = workspace_state.data_view
        xx = x - c * np.cos(theta, dtype=Magic.DataType)
        yy = y - c * np.sin(theta)
        distance = np.sqrt(xx * xx + yy * yy)

        if distance > (a + b) or distance < abs(a - b):
            return []

        q = np.arctan2(yy, xx)
        if distance == (a + b):
            return [State(*wrap_to_pi([q, 0, theta - q]))]
        if distance == abs(a - b):
            return [State(*wrap_to_pi([q, np.pi, theta - q - np.pi]))]

        tmp = (a * a + distance * distance - b * b) / (2 * a * distance)
        q1 = -np.arccos(tmp) + q
        q12 = np.arccos(tmp) + q
        tmp = (a * a + b * b - distance * distance) / (2 * a * b)
        q2 = np.pi - np.arccos(tmp)
        return [
            State(*wrap_to_pi([q1, q2, theta - q1 - q2])),
            State(*wrap_to_pi([q12, -q2, theta - q12 + q2])),
        ]

    def fk(self, s: State) -> State:
        a, b, c = self._link_lengths
        return State(
            a * np.cos(s[0], dtype=Magic.DataType)
            + b * np.cos(s[0] + s[1], dtype=Magic.DataType)
            + c * np.cos(s[0] + s[1] + s[2], dtype=Magic.DataType),
            a * np.sin(s[0], dtype=Magic.DataType)
            + b * np.sin(s[0] + s[1], dtype=Magic.DataType)
            + c * np.sin(s[0] + s[1] + s[2], dtype=Magic.DataType),
            s[0] + s[1] + s[2],
        )


class MoveItCartesianDecomposition(Decomposition):
    """Workspace decomposition that samples via MoveIt FK and IK services."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...], slices: Tuple[int, ...], config: MoveItGroupConfig):
        if config.tool_link_name is None or config.frame_id is None or config.seed_bounds is None:
            raise ValueError("Cartesian decomposition requires tool_link_name, frame_id, and seed_bounds.")

        self._config = config
        super().__init__(lb, ub, slices)
        self.ik_srv = rospy.ServiceProxy(config.ik_service, moveit_msgs.srv.GetPositionIK)
        self.fk_srv = rospy.ServiceProxy(config.fk_service, moveit_msgs.srv.GetPositionFK)

    def set_cell_free_vol(self) -> None:
        for rid, free_volume in self._config.free_volume_overrides.items():
            self._cells_dict[rid].free_vol = free_volume

    def fk(self, s: State) -> State:
        req = moveit_msgs.srv.GetPositionFKRequest()
        req.robot_state.joint_state.name = list(self._config.joint_names)
        req.robot_state.joint_state.position = s.data_view.tolist()
        req.fk_link_names = [self._config.tool_link_name]
        req.header.frame_id = self._config.frame_id
        resp: moveit_msgs.srv.GetPositionFKResponse = self.fk_srv.call(req)

        pose_stamped: geometry_msgs.msg.PoseStamped = resp.pose_stamped[0]
        quaternion = pose_stamped.pose.orientation
        angles = tf.transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return State(
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z,
            *angles,
        )

    def _moveit_ik(self, workspace_state: State, seed: State) -> State | None:
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = self._config.group_name
        req.ik_request.robot_state.joint_state.name = list(self._config.joint_names)
        req.ik_request.robot_state.joint_state.position = seed.data_view.tolist()
        req.ik_request.ik_link_name = self._config.tool_link_name
        req.ik_request.pose_stamped.header.frame_id = self._config.frame_id
        req.ik_request.timeout = rospy.Duration.from_sec(self._config.ik_timeout_sec)
        req.ik_request.pose_stamped.pose.position.x = workspace_state[0]
        req.ik_request.pose_stamped.pose.position.y = workspace_state[1]
        req.ik_request.pose_stamped.pose.position.z = workspace_state[2]

        quaternion = tf.transformations.quaternion_from_euler(
            workspace_state[3], workspace_state[4], workspace_state[5], "sxyz"
        )
        req.ik_request.pose_stamped.pose.orientation.x = quaternion[0]
        req.ik_request.pose_stamped.pose.orientation.y = quaternion[1]
        req.ik_request.pose_stamped.pose.orientation.z = quaternion[2]
        req.ik_request.pose_stamped.pose.orientation.w = quaternion[3]

        resp: moveit_msgs.srv.GetPositionIKResponse = self.ik_srv.call(req)
        if resp.error_code.val == resp.error_code.SUCCESS:
            return State(*resp.solution.joint_state.position)
        return None

    def _sample_in_cell(self, cell: Cell, seed: State | None) -> State | None:
        point = cell.ws.sample_uniform()
        if seed is not None and np.random.random() < self._config.orientation_copy_probability:
            point.data_view[3:] = self.fk(seed).data_view[3:]

        if seed is None:
            lower_bounds, upper_bounds = self._config.seed_bounds
            seed = State(*np.random.uniform(lower_bounds, upper_bounds))

        return self._moveit_ik(point, seed)


__all__ = [
    "MoveItCartesianDecomposition",
    "MoveItGroupConfig",
    "MoveItJointSpace",
    "PlanarDecomposition",
]
