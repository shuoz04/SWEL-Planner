#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/4/19 下午12:18
"""
import rospy
import visualization_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import networkx as nx
import numpy as np
import threading

# %%
rospy.init_node("plot")

# %%
slices = (7, 7, 7)
lb = np.array([-0.8, -0.8, -0.3])
ub = np.array([0.8, 0.8, 1.0])
interval = (ub - lb) / slices
cell_graph: nx.Graph = nx.grid_graph(slices)

# %%
decomp_pub = rospy.Publisher("/vis_decomp", visualization_msgs.msg.MarkerArray, queue_size=1, latch=True)

# %%
msg_array = visualization_msgs.msg.MarkerArray()
msg = visualization_msgs.msg.Marker()
msg.header.frame_id = "right_base_link"
msg.ns = "decomposition"
msg.id = 1
msg.type = msg.CUBE_LIST
msg.action = msg.DELETE
msg.scale.x = interval[0] * 0.95
msg.scale.y = interval[1] * 0.95
msg.scale.z = interval[2] * 0.95
msg.pose.orientation.w = 1.0
msg.color = std_msgs.msg.ColorRGBA(0.25, 0.25, 1.0, 0.2)
msg.lifetime = rospy.Duration.from_sec(0.0)
for rid in cell_graph.nodes:
    _lb = lb + interval * rid
    _ub = _lb + interval
    c = (_lb + _ub) / 2.0
    msg.points.append(geometry_msgs.msg.Point(*c))
msg_array.markers.append(msg)
decomp_pub.publish(msg_array)

# %% draw grid line
msg_array = visualization_msgs.msg.MarkerArray()
msg = visualization_msgs.msg.Marker()
msg.header.frame_id = "right_base_link"
msg.ns = "decomposition"
msg.id = 2
msg.type = msg.LINE_LIST
msg.action = msg.ADD
msg.scale.x = 0.005
msg.pose.orientation.w = 1.0
msg.color = std_msgs.msg.ColorRGBA(0.25, 0.75, 0.6, 0.3)
msg.lifetime = rospy.Duration.from_sec(0.0)

for x in np.linspace(lb[0], ub[0], slices[0] + 1):
    for y in np.linspace(lb[1], ub[1], slices[1] + 1):
        msg.points.append(geometry_msgs.msg.Point(x, y, lb[2]))
        msg.points.append(geometry_msgs.msg.Point(x, y, ub[2]))
    for z in np.linspace(lb[2], ub[2], slices[2] + 1):
        msg.points.append(geometry_msgs.msg.Point(x, lb[1], z))
        msg.points.append(geometry_msgs.msg.Point(x, ub[1], z))
for y in np.linspace(lb[1], ub[1], slices[1] + 1):
    for z in np.linspace(lb[2], ub[2], slices[2] + 1):
        msg.points.append(geometry_msgs.msg.Point(lb[0], y, z))
        msg.points.append(geometry_msgs.msg.Point(ub[0], y, z))
msg_array.markers.append(msg)
decomp_pub.publish(msg_array)

# %%
cell_pub = rospy.Publisher("/vis_cell", visualization_msgs.msg.Marker, queue_size=1, latch=True)

# %%
cell_list = [
    (4, 3, 5),
    (4, 2, 5),
    (4, 2, 4),
    # (3, 2, 4),
    # (3, 1, 4),
]
msg = visualization_msgs.msg.Marker()
msg.header.frame_id = "right_base_link"
msg.ns = "decomposition"
msg.id = 1
msg.type = msg.CUBE_LIST
msg.action = msg.ADD
msg.scale.x = interval[0] * 0.95
msg.scale.y = interval[1] * 0.95
msg.scale.z = interval[2] * 0.95
msg.pose.orientation.w = 1.0
msg.color = std_msgs.msg.ColorRGBA(0.75, 0.25, 1.0, 0.6)
msg.lifetime = rospy.Duration.from_sec(0.0)
for rid in cell_list:
    _lb = lb + interval * rid
    _ub = _lb + interval
    c = (_lb + _ub) / 2.0
    msg.points.append(geometry_msgs.msg.Point(*c))
cell_pub.publish(msg)

# %% draw motion
motion_pub = rospy.Publisher("/vis_motion", visualization_msgs.msg.Marker, queue_size=1, latch=True)

# %% draw motion
msg = visualization_msgs.msg.Marker()
msg.header.frame_id = "right_base_link"
msg.ns = "decomposition"
msg.id = 3
msg.type = msg.LINE_LIST
msg.action = msg.ADD
msg.scale.x = 0.005
msg.pose.orientation.w = 1.0
msg.color = std_msgs.msg.ColorRGBA(0.7, 0.1, 0.7, 0.9)
msg.lifetime = rospy.Duration.from_sec(0.0)
for rid in cell_list:
    _lb = lb + interval * rid
    _ub = _lb + interval
    s = np.random.uniform(_lb, _ub)
    if len(msg.points) >= 2:
        msg.points.append(msg.points[-1])
    msg.points.append(geometry_msgs.msg.Point(*s))
motion_pub.publish(msg)

# %%
joint_pub = rospy.Publisher('yzy_joint_states', sensor_msgs.msg.JointState, queue_size=1, latch=True)

# %%
msg = sensor_msgs.msg.JointState()
msg.name = [f"right_joint_{i}" for i in range(1, 7)]
msg.position = np.deg2rad([0, 120, -90,
                           147, 63, 0]).tolist()
joint_pub.publish(msg)


# %%
def get_joint_state():
    con = threading.Condition()

    def callback(data: sensor_msgs.msg.JointState):
        print(data.name)
        print(np.rad2deg(data.position))
        with con:
            con.notifyAll()

    joint_sub = rospy.Subscriber('joint_states', sensor_msgs.msg.JointState, callback=callback)
    with con:
        con.wait()
        joint_sub.unregister()


# %%
get_joint_state()
