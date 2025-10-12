#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/5/2 下午3:11
"""
import rospy
import numpy as np
import sensor_msgs.msg
import moveit_commander
import moveit_msgs.msg
import trajectory_msgs.msg
from planner.stp import State
import sys
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose, Point, Quaternion
import pandas as pd
# import sys
# from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
#
#
# class MainWindow(QMainWindow):
#     def __init__(self, parent=None, *args, **kwargs):
#         super().__init__(parent, *args, **kwargs)
#         b = QPushButton()
#         b.setText("")
#
#
# if __name__ == '__main__':
#     rospy.init_node("robot_controller")
#     app = QApplication(sys.argv)
#     w = MainWindow(None)
#     w.show()
#     sys.exit(app.exec_())

def create_obstacle_msgs(points, size=0.02):
    # Initialize the ROS node
    rospy.init_node('create_obstacles', anonymous=True)
    
    # Publisher for collision objects
    pub = rospy.Publisher('/move_group/collision_objects', moveit_msgs.msg.CollisionObject, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz
    
    # Create the CollisionObjectArray message
    
    
    for point in points:
        # Create a CollisionObject message for each point
        collision_object = CollisionObject()
        collision_object.id = str(point)  # Using the point coordinates as the ID (not ideal, but for demonstration)
        
        # Define the position and orientation of the obstacle
        pose = Pose()
        pose.position = Point(point[0], point[1], point[2])
        pose.orientation = Quaternion(0, 0, 0, 1)  # Identity quaternion for no rotation
        
        collision_object.pose = pose
        
        # Define the shape and size of the obstacle (a cube)
        size_msg = [size, size, size]  # Cube with side length 'size'
        collision_object.type = 'Box'

        
        # Add the collision object to the array
        pub.publish(collision_object)
    
    # Publish the collision objects
    
    
    # Wait for a while to ensure the message is sent
    rospy.sleep(1)
    
    # Shutdown the node
    rospy.signal_shutdown('Obstacles created and published')

def position_init():
    rospy.init_node("robot_controller")
    pub = rospy.Publisher("custom_joint_states", sensor_msgs.msg.JointState, queue_size=1, latch=True)
    msg = sensor_msgs.msg.JointState()
# # msg.position = np.deg2rad([70, -60, 30])
# msg.position = np.deg2rad([0, 0, 30])
# msg.name = [f"joint_{i}" for i in range(1, 4)]

    mg = moveit_commander.MoveGroupCommander('arm')
    vals = mg.get_current_joint_values()
    print(vals)
    print(np.rad2deg(vals))
# goal = {}
# goal_vals = np.deg2rad([45, 60, -64, 20, -30, 0])
# for i in range(6):
#     goal[f"right_joint_{i + 1}"] = goal_vals[i]
# mg.set_joint_value_target(goal)
    """
    jointvalue: [ 33.297073  , 27.169773   ,-9.3137455  ,43.016502  ,-82.36137 ,60.78329,144.53278  ]
    jointvalue: [ 32.939617 , 27.272095  ,-8.98766   ,43.592518 ,-82.58138 ,  60.732075,145.08644 ]
    jointvalue: [ 33.352734  ,27.220894  ,-9.058283  ,43.164803 ,-82.47955,  60.81722, 144.70715 ]
    jointvalue: [ 32.900158 , 27.359852 , -8.93201  , 43.546307 ,-82.57512   ,60.82835, 145.01907 ]
    jointvalue: [ 33.45029  , 27.048529  ,-9.622079  ,42.947395, -82.2869,    60.735268, 144.42677 ]
    jointvalue: [ 33.384235  ,27.058676 , -9.529316,  43.022583, -82.37575,   60.74724, 144.49681 ]
    jointvalue: [ 33.37908 ,  26.412628 , -8.391195 , 42.789978, -82.72172  , 61.118824 144.0438  ]
    jointvalue: [ 33.462788  ,27.067255 , -9.535419 , 42.87623 , -82.30129 ,  60.777664, 144.36731 ]
    """
    states = [
    State(*np.deg2rad([0, 0, 0, 30, 0, 60, 100])),
    State(*np.deg2rad([ 32.939617 , 27.272095  ,-8.98766   ,43.592518 ,-82.58138 ,  60.732075,145.08644 ]))
    ]
    msg.position = states[0].data_view.tolist()
# msg.position = np.deg2rad([45, 60, -64, 20, -30, 0])

    msg.name = [f"joint{i}" for i in range(1, 8)]

    pub.publish(msg)
    rospy.sleep(0.2)

    trajectory_pub = rospy.Publisher('move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1, latch=True)
    msg = moveit_msgs.msg.DisplayTrajectory()
    msg.model_id = "rm_75"
    traj_msg = moveit_msgs.msg.RobotTrajectory()
    traj_msg.joint_trajectory.joint_names = [f"joint{i}" for i in range(1, 8)]
    for idx, s in enumerate(states):
        pt = trajectory_msgs.msg.JointTrajectoryPoint()
        pt.positions = s.data_view.tolist()
        pt.time_from_start = rospy.Time.from_sec(idx)
        traj_msg.joint_trajectory.points.append(pt)
    msg.trajectory.append(traj_msg)
    msg.trajectory_start.joint_state.name = [f"joint{i}" for i in range(1, 8)]
    msg.trajectory_start.joint_state.position = states[0].data_view.tolist()
    rospy.sleep(0.2)

    trajectory_pub.publish(msg)
    rospy.sleep(0.2)
def read_obs_data(path):
    excel_path = path

    sheet_name = 'Sheet1'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # 确定要读取的行数（不包括标题行）
    num_rows = df.shape[0]  # 这将获取包括标题行的总行数，如果你想要从第一行数据开始，需要减1（如果第一行是标题）

    # 由于你说要读取到x行，但这里的x实际上应该是行数，我们假设你想要读取所有数据行（除了可能的标题行）
    # 如果你有一个特定的x值（行数），你可以将num_rows替换为x（如果x是从0开始的索引，则需要注意+1以匹配实际的行数）
    # 例如：x = 10  # 如果你想读取前10行数据（从第0行开始计数，但通常Excel从第1行开始有数据）
    # num_rows = x + 1  # 如果包括标题行的话；如果不包括，则直接为x

    # 由于我们假设第一行是标题行，所以从第二行开始读取数据
    # 如果你想从第一行开始读取数据作为数据点，则不需要跳过任何行
    data_points = df.to_numpy()
    data_num = data_points.shape[0]
    return data_points
if __name__ == '__main__':
    position_init()
    #data = read_obs_data('/home/zs/0120_catkin_ws/src/mpv2/src/data_of_obs.xlsx')
    #create_obstacle_msgs(data, size=0.02)