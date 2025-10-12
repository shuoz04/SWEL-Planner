#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2021/10/8 下午9:11
"""
from math import pi
import geometry_msgs.msg
import moveit_msgs.msg
import numpy as np
import yaml
from copy import deepcopy
from prettytable import PrettyTable
import tf.transformations
from typing import Union


def rad2deg(rad: Union[float, list, dict, np.ndarray]) -> Union[float, list, dict, np.ndarray]:
    if isinstance(rad, list) or isinstance(rad, tuple):
        return [val / pi * 180 for val in rad]
    elif isinstance(rad, dict):
        tmp = {}
        for key in rad:
            tmp[key] = rad[key] / pi * 180
        return tmp
    else:
        return rad / pi * 180


def deg2rad(deg: Union[float, list, dict, np.ndarray]) -> Union[float, list, dict, np.ndarray]:
    if isinstance(deg, list) or isinstance(deg, tuple):
        return [val / 180 * pi for val in deg]
    elif isinstance(deg, dict):
        tmp = {}
        for key in deg:
            tmp[key] = deg[key] / 180 * pi
        return tmp
    else:
        return deg / 180 * pi


def wrap2pi(rad: float or list) -> float or list:
    if isinstance(rad, list) or isinstance(rad, tuple):
        return [wrap2pi(val) for val in rad]
    else:
        return (rad + pi) % (2 * pi) - pi


class Constant:
    moveit_error_code_map = {
        moveit_msgs.msg.MoveItErrorCodes.SUCCESS: 'success',
        moveit_msgs.msg.MoveItErrorCodes.FAILURE: 'failure',
        moveit_msgs.msg.MoveItErrorCodes.PLANNING_FAILED: 'planning_failed',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_MOTION_PLAN: 'invalid_motion_plan',
        moveit_msgs.msg.MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: 'motion_plan_invalidated_by_environment_change',
        moveit_msgs.msg.MoveItErrorCodes.CONTROL_FAILED: 'control_failed',
        moveit_msgs.msg.MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: 'unable_to_aquire_sensor_data',
        moveit_msgs.msg.MoveItErrorCodes.TIMED_OUT: 'timed_out',
        moveit_msgs.msg.MoveItErrorCodes.PREEMPTED: 'preempted',
        moveit_msgs.msg.MoveItErrorCodes.START_STATE_IN_COLLISION: 'start_state_in_collision',
        moveit_msgs.msg.MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: 'start_state_violates_path_constraints',
        moveit_msgs.msg.MoveItErrorCodes.GOAL_IN_COLLISION: 'goal_in_collision',
        moveit_msgs.msg.MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: 'goal_violates_path_constraints',
        moveit_msgs.msg.MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: 'goal_constraints_violated',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_GROUP_NAME: 'invalid_group_name',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: 'invalid_goal_constraints',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_ROBOT_STATE: 'invalid_robot_state',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_LINK_NAME: 'invalid_link_name',
        moveit_msgs.msg.MoveItErrorCodes.INVALID_OBJECT_NAME: 'invalid_object_name',
        moveit_msgs.msg.MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: 'frame_transform_failure',
        moveit_msgs.msg.MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: 'collision_checking_unavailable',
        moveit_msgs.msg.MoveItErrorCodes.ROBOT_STATE_STALE: 'robot_state_stale',
        moveit_msgs.msg.MoveItErrorCodes.SENSOR_INFO_STALE: 'sensor_info_stale',
        moveit_msgs.msg.MoveItErrorCodes.COMMUNICATION_FAILURE: 'communication_failure',
        moveit_msgs.msg.MoveItErrorCodes.NO_IK_SOLUTION: 'no_ik_solution'
    }


def moveit_error_code_to_string(error_code: moveit_msgs.msg.MoveItErrorCodes) -> str:
    if error_code.val not in Constant.moveit_error_code_map:
        return 'unknown error'
    return Constant.moveit_error_code_map[error_code.val]
