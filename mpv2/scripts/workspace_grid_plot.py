#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/5/10 下午5:58
"""
import open3d as o3d
import rospy
import sensor_msgs.msg
from planner.ColorMapping import convert_numpy_2_pointcloud2_color
import numpy as np

rospy.init_node("pt2", anonymous=True)

# pc: o3d.geometry.PointCloud = o3d.io.read_point_cloud("/home/msi/1008_catkin_ws/src/pickingv2/scripts/data/0714120307_out.ply")
pc: o3d.geometry.PointCloud = o3d.io.read_point_cloud("/home/msi/Documents/ppp/ganzhi_celiang/0000_cloud.pcd")
R = pc.get_rotation_matrix_from_xyz((np.pi, np.pi / 1.5, np.pi / 2))
pc = pc.rotate(R, center=(0, 0, 0))

pub = rospy.Publisher("/cloud_in", sensor_msgs.msg.PointCloud2, queue_size=3)

pts = np.asarray(pc.points, dtype=np.float32)
# print(pts)
print(pts.shape)
print(pts.dtype)
print(pc.points)
# pts = pts / 1000.0

# pts[:, -1] -= 0.6
# pts[:, 1] -= 0.75
pts[:, 0] += 0.3
idx = np.where(pts[:, 0] < 1.0)
colors = np.asarray(pc.colors)

tomatoes = np.array([
    [0.0, -0.3, 0.7],
    [0.2, -0.21, 0.45],
    [0.1, -0.23, 0.63],
    # [0.67, -0.23, 0.58],
    # [0.62, -0.23, 0.63],
    # [0.65, -0.36, 0.6],
], dtype=np.float32)

tomato_pub = rospy.Publisher("/tomato", sensor_msgs.msg.PointCloud2, queue_size=3)

c = 1000
r = rospy.Rate(10)
while not rospy.is_shutdown():
    msg = convert_numpy_2_pointcloud2_color(pts[idx], colors[idx], frame_id="camera_link", maxDistColor=2)
    pub.publish(msg)

    # msg2 = convert_numpy_2_pointcloud2_color(tomatoes, frame_id="platform")
    # tomato_pub.publish(msg2)

    r.sleep()
    # c -= 1
