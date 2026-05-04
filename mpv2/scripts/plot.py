#!/usr/bin/env python3
"""ROS visualization helper for workspace decomposition debugging."""

from __future__ import annotations

import threading

import geometry_msgs.msg
import networkx as nx
import numpy as np
import rospy
import sensor_msgs.msg
import std_msgs.msg
import visualization_msgs.msg


SLICES = (7, 7, 7)
LOWER_BOUNDS = np.array([-0.8, -0.8, -0.3], dtype=np.float64)
UPPER_BOUNDS = np.array([0.8, 0.8, 1.0], dtype=np.float64)
CELL_LIST = [
    (4, 3, 5),
    (4, 2, 5),
    (4, 2, 4),
]
DEFAULT_JOINT_POSITION = np.deg2rad([0, 120, -90, 147, 63, 0]).tolist()


def _interval() -> np.ndarray:
    return (UPPER_BOUNDS - LOWER_BOUNDS) / SLICES


def _cell_center(rid: tuple[int, int, int]) -> np.ndarray:
    interval = _interval()
    lower = LOWER_BOUNDS + interval * rid
    upper = lower + interval
    return (lower + upper) / 2.0


def _build_marker(marker_id: int, marker_type: int, scale_xyz: tuple[float, float, float]) -> visualization_msgs.msg.Marker:
    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = "right_base_link"
    marker.ns = "decomposition"
    marker.id = marker_id
    marker.type = marker_type
    marker.scale.x, marker.scale.y, marker.scale.z = scale_xyz
    marker.pose.orientation.w = 1.0
    marker.lifetime = rospy.Duration.from_sec(0.0)
    return marker


def publish_decomposition(publisher: rospy.Publisher) -> None:
    cell_graph: nx.Graph = nx.grid_graph(SLICES)
    interval = _interval()

    marker_array = visualization_msgs.msg.MarkerArray()
    marker = _build_marker(1, visualization_msgs.msg.Marker.CUBE_LIST, (interval[0] * 0.95, interval[1] * 0.95, interval[2] * 0.95))
    marker.action = visualization_msgs.msg.Marker.DELETE
    marker.color = std_msgs.msg.ColorRGBA(0.25, 0.25, 1.0, 0.2)

    for rid in cell_graph.nodes:
        marker.points.append(geometry_msgs.msg.Point(*_cell_center(rid)))

    marker_array.markers.append(marker)
    publisher.publish(marker_array)


def publish_grid_lines(publisher: rospy.Publisher) -> None:
    marker_array = visualization_msgs.msg.MarkerArray()
    marker = _build_marker(2, visualization_msgs.msg.Marker.LINE_LIST, (0.005, 0.0, 0.0))
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.color = std_msgs.msg.ColorRGBA(0.25, 0.75, 0.6, 0.3)

    for x in np.linspace(LOWER_BOUNDS[0], UPPER_BOUNDS[0], SLICES[0] + 1):
        for y in np.linspace(LOWER_BOUNDS[1], UPPER_BOUNDS[1], SLICES[1] + 1):
            marker.points.append(geometry_msgs.msg.Point(x, y, LOWER_BOUNDS[2]))
            marker.points.append(geometry_msgs.msg.Point(x, y, UPPER_BOUNDS[2]))
        for z in np.linspace(LOWER_BOUNDS[2], UPPER_BOUNDS[2], SLICES[2] + 1):
            marker.points.append(geometry_msgs.msg.Point(x, LOWER_BOUNDS[1], z))
            marker.points.append(geometry_msgs.msg.Point(x, UPPER_BOUNDS[1], z))

    for y in np.linspace(LOWER_BOUNDS[1], UPPER_BOUNDS[1], SLICES[1] + 1):
        for z in np.linspace(LOWER_BOUNDS[2], UPPER_BOUNDS[2], SLICES[2] + 1):
            marker.points.append(geometry_msgs.msg.Point(LOWER_BOUNDS[0], y, z))
            marker.points.append(geometry_msgs.msg.Point(UPPER_BOUNDS[0], y, z))

    marker_array.markers.append(marker)
    publisher.publish(marker_array)


def publish_cells(publisher: rospy.Publisher) -> None:
    interval = _interval()
    marker = _build_marker(1, visualization_msgs.msg.Marker.CUBE_LIST, (interval[0] * 0.95, interval[1] * 0.95, interval[2] * 0.95))
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.color = std_msgs.msg.ColorRGBA(0.75, 0.25, 1.0, 0.6)

    for rid in CELL_LIST:
        marker.points.append(geometry_msgs.msg.Point(*_cell_center(rid)))

    publisher.publish(marker)


def publish_motion(publisher: rospy.Publisher) -> None:
    interval = _interval()
    marker = _build_marker(3, visualization_msgs.msg.Marker.LINE_LIST, (0.005, 0.0, 0.0))
    marker.action = visualization_msgs.msg.Marker.ADD
    marker.color = std_msgs.msg.ColorRGBA(0.7, 0.1, 0.7, 0.9)

    for rid in CELL_LIST:
        lower = LOWER_BOUNDS + interval * rid
        upper = lower + interval
        sample = np.random.uniform(lower, upper)
        if len(marker.points) >= 2:
            marker.points.append(marker.points[-1])
        marker.points.append(geometry_msgs.msg.Point(*sample))

    publisher.publish(marker)


def publish_joint_state(publisher: rospy.Publisher) -> None:
    message = sensor_msgs.msg.JointState()
    message.name = [f"right_joint_{i}" for i in range(1, 7)]
    message.position = DEFAULT_JOINT_POSITION
    publisher.publish(message)


def get_joint_state() -> None:
    condition = threading.Condition()

    def callback(data: sensor_msgs.msg.JointState) -> None:
        print(data.name)
        print(np.rad2deg(data.position))
        with condition:
            condition.notify_all()

    subscriber = rospy.Subscriber("joint_states", sensor_msgs.msg.JointState, callback=callback)
    with condition:
        condition.wait()
        subscriber.unregister()


def main() -> None:
    rospy.init_node("plot")

    decomp_publisher = rospy.Publisher("/vis_decomp", visualization_msgs.msg.MarkerArray, queue_size=1, latch=True)
    cell_publisher = rospy.Publisher("/vis_cell", visualization_msgs.msg.Marker, queue_size=1, latch=True)
    motion_publisher = rospy.Publisher("/vis_motion", visualization_msgs.msg.Marker, queue_size=1, latch=True)
    joint_publisher = rospy.Publisher("yzy_joint_states", sensor_msgs.msg.JointState, queue_size=1, latch=True)

    publish_decomposition(decomp_publisher)
    publish_grid_lines(decomp_publisher)
    publish_cells(cell_publisher)
    publish_motion(motion_publisher)
    publish_joint_state(joint_publisher)
    get_joint_state()


if __name__ == "__main__":
    main()
