#!/usr/bin/env python3
"""Publish a point cloud file to ROS for workspace debugging."""

from __future__ import annotations

import open3d as o3d
import numpy as np
import rospy
import sensor_msgs.msg

from planner.ColorMapping import convert_numpy_2_pointcloud2_color


POINT_CLOUD_PATH = "/home/msi/Documents/ppp/ganzhi_celiang/0000_cloud.pcd"
ROTATION_EULER = (np.pi, np.pi / 1.5, np.pi / 2)
POINT_OFFSET = np.array([0.3, 0.0, 0.0], dtype=np.float32)
X_LIMIT = 1.0
TOMATO_POINTS = np.array(
    [
        [0.0, -0.3, 0.7],
        [0.2, -0.21, 0.45],
        [0.1, -0.23, 0.63],
    ],
    dtype=np.float32,
)


def load_point_cloud(path: str = POINT_CLOUD_PATH) -> tuple[np.ndarray, np.ndarray]:
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(path)
    rotation = point_cloud.get_rotation_matrix_from_xyz(ROTATION_EULER)
    point_cloud = point_cloud.rotate(rotation, center=(0, 0, 0))

    points = np.asarray(point_cloud.points, dtype=np.float32)
    colors = np.asarray(point_cloud.colors)
    points[:, 0] += POINT_OFFSET[0]
    return points, colors


def publish_workspace_cloud() -> None:
    rospy.init_node("pt2", anonymous=True)

    cloud_publisher = rospy.Publisher("/cloud_in", sensor_msgs.msg.PointCloud2, queue_size=3)
    tomato_publisher = rospy.Publisher("/tomato", sensor_msgs.msg.PointCloud2, queue_size=3)
    points, colors = load_point_cloud()
    selected_indices = np.where(points[:, 0] < X_LIMIT)

    print(points.shape)
    print(points.dtype)
    print(points)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        message = convert_numpy_2_pointcloud2_color(
            points[selected_indices],
            colors[selected_indices],
            frame_id="camera_link",
            maxDistColor=2,
        )
        cloud_publisher.publish(message)

        # Keep the publisher around for parity with the original script.
        _ = tomato_publisher
        _ = TOMATO_POINTS
        rate.sleep()


if __name__ == "__main__":
    publish_workspace_cloud()
