#!/usr/bin/env python3
"""Utilities for publishing debug robot state and obstacle data."""

from __future__ import annotations

from typing import Sequence

import moveit_commander
import moveit_msgs.msg
import numpy as np
import pandas as pd
import rospy
import sensor_msgs.msg
import trajectory_msgs.msg
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject

from planner.stp import State


RM75_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
DEFAULT_STATES = (
    State(*np.deg2rad([0, 0, 0, 30, 0, 60, 100])),
    State(*np.deg2rad([32.939617, 27.272095, -8.98766, 43.592518, -82.58138, 60.732075, 145.08644])),
)


def _ensure_node(name: str, anonymous: bool = False) -> None:
    if not rospy.core.is_initialized():
        rospy.init_node(name, anonymous=anonymous)


def create_obstacle_msgs(points: Sequence[Sequence[float]], size: float = 0.02) -> None:
    """Publish legacy collision objects for a list of point samples."""
    _ensure_node("create_obstacles", anonymous=True)
    publisher = rospy.Publisher("/move_group/collision_objects", moveit_msgs.msg.CollisionObject, queue_size=10)

    for point in points:
        collision_object = CollisionObject()
        collision_object.id = str(tuple(point))

        pose = Pose()
        pose.position = Point(point[0], point[1], point[2])
        pose.orientation = Quaternion(0, 0, 0, 1)
        collision_object.pose = pose
        collision_object.type = "Box"

        # ``size`` is kept for API compatibility with the original helper.
        _ = size
        publisher.publish(collision_object)

    rospy.sleep(1)
    rospy.signal_shutdown("Obstacles created and published")


def _build_joint_state_message(state: State) -> sensor_msgs.msg.JointState:
    message = sensor_msgs.msg.JointState()
    message.name = list(RM75_JOINT_NAMES)
    message.position = state.data_view.tolist()
    return message


def _build_display_trajectory(states: Sequence[State]) -> moveit_msgs.msg.DisplayTrajectory:
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.model_id = "rm_75"
    display_trajectory.trajectory_start.joint_state.name = list(RM75_JOINT_NAMES)
    display_trajectory.trajectory_start.joint_state.position = states[0].data_view.tolist()

    robot_trajectory = moveit_msgs.msg.RobotTrajectory()
    robot_trajectory.joint_trajectory.joint_names = list(RM75_JOINT_NAMES)

    for index, state in enumerate(states):
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions = state.data_view.tolist()
        point.time_from_start = rospy.Time.from_sec(index)
        robot_trajectory.joint_trajectory.points.append(point)

    display_trajectory.trajectory.append(robot_trajectory)
    return display_trajectory


def position_init(states: Sequence[State] = DEFAULT_STATES) -> None:
    """Publish a deterministic debug trajectory for the RM75 arm."""
    _ensure_node("robot_controller")

    move_group = moveit_commander.MoveGroupCommander("arm")
    current_joint_values = move_group.get_current_joint_values()
    print(current_joint_values)
    print(np.rad2deg(current_joint_values))

    joint_state_publisher = rospy.Publisher("custom_joint_states", sensor_msgs.msg.JointState, queue_size=1, latch=True)
    joint_state_publisher.publish(_build_joint_state_message(states[0]))
    rospy.sleep(0.2)

    trajectory_publisher = rospy.Publisher(
        "move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=1,
        latch=True,
    )
    trajectory_publisher.publish(_build_display_trajectory(states))
    rospy.sleep(0.2)


def read_obs_data(path: str, sheet_name: str = "Sheet1") -> np.ndarray:
    """Load obstacle samples from an Excel worksheet."""
    return pd.read_excel(path, sheet_name=sheet_name).to_numpy()


if __name__ == "__main__":
    position_init()
