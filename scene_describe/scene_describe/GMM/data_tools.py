import time

import numpy as np
from pybullet_utils import bullet_client

import os
from sklearn.preprocessing import MinMaxScaler
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from scipy.spatial.transform import Rotation as R
import numpy as np
import ast

import numpy as np

import pybullet as p

import pybullet_data

from pybullet_utils import bullet_client
import numpy as np


def data_read(file_path):
    data = []
    # 读取文件并按组合并每两行
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            # 去掉方括号并去除每行的换行符
            line1 = list(map(float, lines[i].strip()[1:-1].split()))
            line2 = list(map(float, lines[i + 1].strip()[1:-1].split()))
            # 将前六个维度和第七个维度合并成一行
            combined_line = line1 + line2
            data.append(combined_line)

    # 转换为numpy数组
    np_array = np.array(data)
    return np_array

def data_process(data,gui = True):
    if gui == False:
        selfp = bullet_client.BulletClient(connection_mode=p.DIRECT)
    else:
        selfp = bullet_client.BulletClient(connection_mode=p.GUI)
    # self.p.setTimeStep(1/240)
    # print(self.p)
    selfp.setGravity(0, 0, -9.81)

    selfarm = selfp.loadURDF(
        "D:\\FR5_Reinforcement-learning-master\\jaka_rl_subgoal_SAC\\robot_model\\rm_description\\urdf\\RM75\\rm_75.urdf",
        useFixedBase=True, basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_SELF_COLLISION)
    p.stepSimulation()
    orien = []
    pos = []
    for item in data:
        for i in range(7):
            selfp.resetJointState(selfarm, i, item[i])
            selfp.stepSimulation()
        gripper_orien = p.getLinkState(selfarm, 7)[5]
        # gripper_orien = [gripper_orien[3],gripper_orien[0],gripper_orien[1],gripper_orien[2]]
        gripper_pos = p.getLinkState(selfarm, 7)[4]
        pos.append(gripper_pos)
        orien.append(gripper_orien)
        # time.sleep(0.5)
    orien = np.array(orien,dtype= np.float32)
    pos = np.array(pos,dtype= np.float32)
    # 创建MinMaxScaler对象，将数据归一化到[0, 1]区间
    norms_y = np.linalg.norm(orien, axis=1, keepdims=True)  # 计算每行的 L2 范数
    data = orien / norms_y  # 每行除以其范数

    return data,pos

def getWeightsforData(data,gui = True):
    if gui == False:
        selfp = bullet_client.BulletClient(connection_mode=p.DIRECT)
    else:
        selfp = bullet_client.BulletClient(connection_mode=p.GUI)
    # self.p.setTimeStep(1/240)
    # print(self.p)
    selfp.setGravity(0, 0, -9.81)

    selfarm = selfp.loadURDF(
        "D:\\FR5_Reinforcement-learning-master\\jaka_rl_subgoal_SAC\\robot_model\\rm_description\\urdf\\RM75\\rm_75.urdf",
        useFixedBase=True, basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_SELF_COLLISION)
    p.stepSimulation()
    angle = []
    weights = []
    for item in data:
        for i in range(7):
            selfp.resetJointState(selfarm, i, item[i])
            selfp.stepSimulation()
        gripper_orien = p.getLinkState(selfarm, 7)[5]
        gripper_z = np.array([0, 0, 1])
        gripper_orientation = R.from_quat(gripper_orien)
        gripper_z_rot = gripper_orientation.apply(gripper_z)
        target_orientation = np.array( [ 0.15496272 , 0.21455847 ,0,         0.9643398 ], dtype=np.float32)
        target_rotate = R.from_quat(target_orientation)
        target_orien = target_rotate.apply([0, 0, 1])
        # 计算target_orien和gripper_z_rot的夹角
        a = np.array(gripper_z_rot)
        b = np.array(target_orien)
        # 计算模（或范数）
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        dot = np.dot(a, b)
        if dot < 0:
            a = -a
            dot = np.dot(a, b)
        # 计算cos(θ)
        cos_theta = dot / (magnitude_a * magnitude_b)

        angle_radians = np.arccos(cos_theta)
        angle.append(angle_radians)
        gripper_pos = p.getLinkState(selfarm, 7)[4]
        rotation = R.from_quat(p.getLinkState(selfarm, 7)[5])
        point_befor_rotate = np.array([-0.01, 0.07, 0.19])
        point_after_rotate = rotation.apply(point_befor_rotate)
        end_pos = point_after_rotate + np.array(gripper_pos)
        dis = np.array([0.6,0,0.65]) - end_pos
        dis = np.linalg.norm(dis)
        if dis > 0.03:
            weight = 0.1
        if dis <= 0.03:
            weight = dis / 0.03

        weights.append(weight)
    weights = np.array(weights,dtype= np.float32)
    angle = np.array(angle,dtype= np.float32)
    return weights,angle
    # 返回弧度制角度

def evaluate_sample_data(oriendata):
    weights = []
    for item in oriendata:
        gripper_orien = item
        gripper_z = np.array([0, 0, 1])
        gripper_orientation = R.from_quat(gripper_orien)
        gripper_z_rot = gripper_orientation.apply(gripper_z)
        target_orientation = np.array([ 0.15496272 , 0.21455847 ,0,         0.9643398 ], dtype=np.float32)
        target_rotate = R.from_quat(target_orientation)
        target_orien = target_rotate.apply([0, 0, 1])
        # 计算target_orien和gripper_z_rot的夹角
        a = np.array(gripper_z_rot)
        b = np.array([0.40662378,0.35998749,0.83968])
        # 计算模（或范数）
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        dot = np.dot(a, b)
        if dot < 0:
            a = -a
            dot = np.dot(a, b)
        # 计算cos(θ)
        cos_theta = dot / (magnitude_a * magnitude_b)
        angle_radians = np.arccos(cos_theta)
        weight = angle_radians/(0.5*np.pi)

        weights.append(weight)
    weights = np.array(weights,dtype= np.float32)
    return weights

def read_weights(path):
    last_values = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # 按 "])," 分割，得到的第二部分应该包含最后一个数字
            parts = line.split("]),")
            if len(parts) == 2:
                # 去掉第二部分中可能的']'和空白字符
                num_str = parts[1].strip(" ]")
                try:
                    value = float(num_str)
                    last_values.append(value)
                except ValueError:
                    print("转换数字失败:", num_str)
            else:
                print("格式异常的行:", line)

    result_array = np.array(last_values)
    return result_array




if __name__ == "__main__":
    # path = "./dataset/data.txt"
    # data = data_read(path)
    # data_ = data_process(data)
    # weights,angle = getWeightsforData(data)
    # print(weights)
    # 假设你的数据文件名为 "data.txt"

    path = "./data/sampling-data/pos/pos_10%.txt"
    last_values = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # 按 "])," 分割，得到的第二部分应该包含最后一个数字
            parts = line.split("]),")
            if len(parts) == 2:
                # 去掉第二部分中可能的']'和空白字符
                num_str = parts[1].strip(" ]")
                try:
                    value = float(num_str)
                    last_values.append(value)
                except ValueError:
                    print("转换数字失败:", num_str)
            else:
                print("格式异常的行:", line)

    result_array = np.array(last_values)
    print(result_array)

    # results = read_weights(path)
    # print(results)