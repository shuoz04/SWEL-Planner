#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/4/18 下午12:36
"""
import time
from typing import Union, Tuple, List, Dict
import sys
import moveit_commander
import numpy as np
import rospy

from planner.stp import STP, Space, Decomposition, State, Magic, wrap_to_pi, Cell

import moveit_msgs.srv
import geometry_msgs.msg
import tf.transformations
import os
from robot_controller import position_init
"""
define the working space and a method of check the validity of the robot state 
"""
os.environ['OMP_NUM_THREADS'] = '2'
def weighted_quaternion_mean(q_list, weights, max_iter=20, eps=1e-6):
    """计算加权平均四元数"""
    q_ref = np.asarray(q_list[0], dtype=np.float64)  # 参考四元数，确保是 numpy 数组
    q_ref /= np.linalg.norm(q_ref)  # 单位化

    for _ in range(max_iter):
        v_sum = np.zeros(3)
        for q, w in zip(q_list, weights):
            q = np.asarray(q, dtype=np.float64)  # 确保 q 是 numpy 数组
            v = log_map(q, q_ref)  # 计算切空间向量
            v_sum += w * v  # 加权求和
        # print("v_sum:", v_sum)
        delta_q = exp_map(v_sum, q_ref)
        # print("delta_q:", delta_q)
        # print("q_ref:", q_ref)
        q_ref = quaternion_multiply(delta_q, q_ref)  # 更新参考四元数
        q_ref /= np.linalg.norm(q_ref)  # 单位化，确保 q_ref 始终是单位四元数
    return q_ref


def log_map(q, q_ref):
    """四元数到切空间映射"""
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    q_inv = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])  # 计算 q_ref 的共轭
    q_diff = quaternion_multiply(q, q_inv)  # 计算 q * q_ref^(-1)

    theta = np.arccos(np.clip(q_diff[0], -1, 1))  # 计算旋转角
    if theta < 1e-6:
        return np.zeros(3)  # 角度很小时返回零向量

    return (theta / np.sin(theta)) * q_diff[1:]  # 提取虚部


def exp_map(v, q_ref):
    """切空间到四元数逆映射"""
    v = np.asarray(v, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    theta = np.linalg.norm(v)  # 旋转角度
    if theta < 1e-6:
        return q_ref.copy()  # 零向量时返回参考四元数

    q_exp = np.concatenate([[np.cos(theta)], (np.sin(theta)/theta) * v])  # 指数映射
    return quaternion_multiply(q_exp, q_ref)  # 确保返回四元数



def quaternion_multiply(q1, q2):
    """Hamilton 乘法，计算 q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # 实部
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # i
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # j
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # k
    ])

class WeightedGMM:
    def __init__(self, n_components, shared_cov=False, max_iter=100, tol=1e-4):
        self.K = n_components
        self.shared_cov = shared_cov
        self.max_iter = max_iter
        self.tol = tol
        self.q_ref = []

    def fit(self, V, weights):
        """V: 切空间向量 (N x 3)
           weights: 样本权重 (N,) """
        N, d = V.shape
        # 初始化参数
        kmeans = KMeans(n_clusters=self.K, n_init=10).fit(V)
        self.pi = np.bincount(kmeans.labels_, weights=weights) / np.sum(weights)
        self.mu = kmeans.cluster_centers_
        if self.shared_cov:
            self.Sigma = np.cov(V.T, aweights=weights)
            self.Sigma = np.stack([self.Sigma] * self.K, axis=0)
        else:
            self.Sigma = np.array([np.cov(V[kmeans.labels_ == k].T,
                                          aweights=weights[kmeans.labels_ == k])
                                   for k in range(self.K)])

        # EM迭代
        prev_loglik = -np.inf
        for it in range(self.max_iter):
            # E-Step
            gamma = self._e_step(V, weights)

            # M-Step
            self._m_step(V, weights, gamma)

            # 计算对数似然
            loglik = self._log_likelihood(V, weights)
            if np.abs(loglik - prev_loglik) < self.tol:
                break
            prev_loglik = loglik

    def _e_step(self, V, weights):
        eps = 1e-8
        gamma = np.zeros((len(V), self.K))
        for k in range(self.K):
            diff = V - self.mu[k]
            cov = self.Sigma[k] + 1e-6 * np.eye(3)  # 正则化
            inv_cov = np.linalg.inv(cov)
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            det_cov = np.linalg.det(cov) + eps
            gamma[:, k] = self.pi[k] * np.exp(exp_term) / np.sqrt(det_cov)

            #gamma[:, k] = self.pi[k] * np.exp(exp_term) / np.sqrt(np.linalg.det(2 * np.pi * cov))
        gamma *= weights[:, None]

        gamma_sum = gamma.sum(axis=1, keepdims=True) + eps
        gamma /= gamma_sum

        #gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, V, weights, gamma):
        N_k = gamma.sum(axis=0)
        eps = 1e-8
        self.pi = (N_k + eps) / (N_k.sum() + eps)

        #self.pi = N_k / N_k.sum()

        for k in range(self.K):
            self.mu[k] = np.sum(gamma[:, k][:, None] * weights[:, None] * V, axis=0) / (gamma[:, k] * weights).sum()

        if self.shared_cov:
            cov = np.zeros((3, 3))
            for k in range(self.K):
                diff = V - self.mu[k]
                cov += (gamma[:, k, None, None] * weights[:, None, None] *
                        np.einsum('ni,nj->nij', diff, diff)).sum(axis=0)
            self.Sigma = cov / N_k.sum()
            self.Sigma = np.stack([self.Sigma] * self.K, axis=0)
        else:
            for k in range(self.K):
                diff = V - self.mu[k]
                self.Sigma[k] = (gamma[:, k, None, None] * weights[:, None, None] *
                                 np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / (N_k[k] + eps)

                # self.Sigma[k] = (gamma[:, k, None, None] * weights[:, None, None] *
                #                  np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / N_k[k]

    def sample(self, n_samples):
        k = np.random.choice(self.K, p=self.pi, size=n_samples)
        samples = []
        for ki in k:
            v = np.random.multivariate_normal(self.mu[ki], self.Sigma[ki])
            samples.append(v)
        return np.array(samples)

    def _log_likelihood(self, V, weights):
        """计算加权对数似然"""
        loglik = 0
        for k in range(self.K):
            diff = V - self.mu[k]
            cov = self.Sigma[k] + 1e-6 * np.eye(3)  # 加上小正则项避免数值问题
            inv_cov = np.linalg.inv(cov)
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            loglik += np.sum(weights * (np.log(self.pi[k]) + exp_term - 0.5 * np.log(np.linalg.det(2 * np.pi * cov))))
        return loglik










class PlanarSpace(Space):
    def __init__(self, lb: Tuple, ub: Tuple):
        super().__init__(lb, ub)
        self.check_validity_srv = rospy.ServiceProxy('/check_state_validity', moveit_msgs.srv.GetStateValidity)

    def check_validity(self, s: State) -> bool:
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = 'arm'
        req.robot_state.joint_state.name = ['joint_1', 'joint_2', 'joint_3']
        req.robot_state.joint_state.position = s.data_view.tolist()
        resp: moveit_msgs.srv.GetStateValidityResponse = self.check_validity_srv.call(req)
        return resp.valid


class PlanarDecomp(Decomposition):
    @staticmethod
    def _set_cell_free_vol(cells_dict: Dict[Tuple, Cell]):
        # cells_dict[(7, 4)].free_vol = 0.1
        cells_dict[(0, 0)].free_vol = 0.1

    def _sample_in_cell(self, cell: Cell, seed: None or State) -> Union[State, None]:
        pt = cell.ws.sample_uniform().data_view
        if seed is not None and np.random.random() < 0.9:
            pt[-1] = self.fk(seed).data_view[-1]
        states = self._aik(State(*pt))
        if not states:
            return None
        if seed is None:
            return states[np.random.randint(0, len(states))]
        else:
            return min(states, key=lambda _s: np.linalg.norm(_s.data_view[:2] - seed.data_view[:2]))

    @staticmethod
    def _aik(ws_s: State) -> List[State]:
        a, b, c = 0.5, 0.4, 0.1
        x, y, theta = ws_s.data_view
        xx = x - c * np.cos(theta, dtype=Magic.DataType)
        yy = y - c * np.sin(theta)
        d = np.sqrt(xx * xx + yy * yy)
        if d > (a + b) or d < abs(a - b):
            return []
        q = np.arctan2(yy, xx)
        if d == (a + b):
            return [State(*wrap_to_pi([q, 0, theta - q])), ]
        if d == abs(a - b):
            return [State(*wrap_to_pi([q, np.pi, theta - q - np.pi])), ]

        tmp = (a * a + d * d - b * b) / (2 * a * d)
        q1 = -np.arccos(tmp) + q
        q12 = np.arccos(tmp) + q
        tmp = (a * a + b * b - d * d) / (2 * a * b)
        q2 = np.pi - np.arccos(tmp)
        return [State(*wrap_to_pi([q1, q2, theta - q1 - q2])), State(*wrap_to_pi([q12, -q2, theta - q12 + q2])), ]

    def fk(self, s: State) -> State:
        return State(
            0.5 * np.cos(s[0], dtype=Magic.DataType) + 0.4 * np.cos(s[0] + s[1], dtype=Magic.DataType) + 0.1 * np.cos(
                s[0] + s[1] + s[2], dtype=Magic.DataType),
            0.5 * np.sin(s[0], dtype=Magic.DataType) + 0.4 * np.sin(s[0] + s[1], dtype=Magic.DataType) + 0.1 * np.sin(
                s[0] + s[1] + s[2], dtype=Magic.DataType),
            s[0] + s[1] + s[2]
        )


class JakaSpace(Space):
    def __init__(self, lb: Tuple, ub: Tuple):
        super().__init__(lb, ub)
        self.check_validity_srv = rospy.ServiceProxy('/check_state_validity', moveit_msgs.srv.GetStateValidity)

    def check_validity(self, s: State) -> bool:
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = 'arm'
        req.robot_state.joint_state.name = [f'joint{i}' for i in range(1, 8)]
        req.robot_state.joint_state.position = s.data_view.tolist()
        resp: moveit_msgs.srv.GetStateValidityResponse = self.check_validity_srv.call(req)
        return resp.valid

    
class JakaDecomp(Decomposition):
    def __init__(self, lb: Tuple, ub: Tuple, slices: Tuple):
        super().__init__(lb, ub, slices)
        self.ik_srv = rospy.ServiceProxy("/compute_ik", moveit_msgs.srv.GetPositionIK)
        self.fk_srv = rospy.ServiceProxy("/compute_fk", moveit_msgs.srv.GetPositionFK)

    @staticmethod
    def _set_cell_free_vol(cells_dict: Dict[Tuple, Cell]):
        pass
        # cells_dict[(8, 6, 8)].free_vol = 0.2
        cells_dict[(0, 0, 0)].free_vol = 0.2


    def fk(self, s: State) -> State:
        req = moveit_msgs.srv.GetPositionFKRequest()
        req.robot_state.joint_state.name = [f'joint{i}' for i in range(1, 8)]
        req.robot_state.joint_state.position = s.data_view.tolist()
        req.fk_link_names = ["tool_link"]
        req.header.frame_id = "base_link"
        resp: moveit_msgs.srv.GetPositionFKResponse = self.fk_srv.call(req)
        pose_stamped: geometry_msgs.msg.PoseStamped = resp.pose_stamped[0]
        qua = pose_stamped.pose.orientation
        angles = tf.transformations.euler_from_quaternion([
            qua.x, qua.y, qua.z, qua.w
        ])
        return State(pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z, *angles)

    def _moveit_ik(self, w_s: State, seed: State) -> Union[State, None]:
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = 'arm'
        req.ik_request.robot_state.joint_state.name = [f'right_joint_{i}' for i in range(1, 7)]
        req.ik_request.robot_state.joint_state.position = seed.data_view.tolist()  #seed在此处其作用，即告诉求解器：“请从这些关节角度开始搜索解”
        """
        然而，如果 seed 提供的关节角度与实际解相差甚远，或者根本不在有效的关节空间内，那么求解器可能会花费更长时间来找到解，或者可能无法找到解。

        因此，在调用逆运动学求解函数时，合理地选择 seed 参数是非常重要的。这通常需要根据机器人的当前状态、末端执行器的目标位置以及机器人的运动学特性来做出决策。在实际应用中，可能会使用机器人的当前关节角度作为 seed，或者根据之前的运动规划结果来预测一个合理的起始点。
        """
        req.ik_request.ik_link_name = "tool_link"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        req.ik_request.timeout = rospy.Duration.from_sec(1.0)
        req.ik_request.pose_stamped.pose.position.x = w_s[0]
        req.ik_request.pose_stamped.pose.position.y = w_s[1]
        req.ik_request.pose_stamped.pose.position.z = w_s[2]
        qua = tf.transformations.quaternion_from_euler(w_s[3], w_s[4], w_s[5], 'sxyz')
        req.ik_request.pose_stamped.pose.orientation.x = qua[0]
        req.ik_request.pose_stamped.pose.orientation.y = qua[1]
        req.ik_request.pose_stamped.pose.orientation.z = qua[2]
        req.ik_request.pose_stamped.pose.orientation.w = qua[3]
        resp: moveit_msgs.srv.GetPositionIKResponse = self.ik_srv.call(req)
        if resp.error_code.val == resp.error_code.SUCCESS:
            return State(*resp.solution.joint_state.position)
    

    def _sample_in_cell(self, cell: Cell, seed: Union[None, State]) -> Union[State, None]:
        pt = cell.ws.sample_uniform()     #在workspace中随机采样一个点作为pt（6维度）
        if seed is not None and np.random.random() < 0.95:#np.random.random返回一个在0到1之间随机均匀的浮点数
            pt.data_view[3:] = self.fk(seed).data_view[3:]  #给pt赋予姿态，他的姿态来源于seed的姿态
        if seed is None:
            seed = State(*np.random.uniform(    #均匀采样
        (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
        (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi)
            ))              #关节空间限制下的随机采样

        return self._moveit_ik(pt, seed)#返回了一个逆解之后的6dim结果
    def ik_from_potential(self,point,seed):
        orien = np.array(self.fk(seed).data_view[3:])
        data_vie = np.hstack((point,orien))
        s = State(*data_vie)

        s.data_view[3:] = self.fk(seed).data_view[3:]
        if self._moveit_ik(s, seed):
            return self._moveit_ik(s, seed)
        else: return 0

def planar():
    space = PlanarSpace((-np.pi, -np.pi, -np.pi), (np.pi, np.pi, np.pi))
    decomp = PlanarDecomp((-1.1, -1.1, -np.pi), (1.1, 1.1, np.pi), (9, 7))

    stp = STP(space, decomp)

    s1 = State(*np.deg2rad([0, 0, 30]))
    s2 = State(*np.deg2rad([120, -60, 30]))

    t = time.time()
    res = stp.solve(s1, s2)
    cost = time.time() - t
    print(f"res: {res}")
    print(f"cost: {cost * 1000.0:.2f} ms")


def jaka():
    # robot = moveit_commander.RobotCommander()
    # for joint_name in robot.get_joint_names('arm'):
    #     j: moveit_commander.RobotCommander.Joint = robot.get_joint(joint_name)
    #     # print(f"{joint_name}: {j.min_bound()} ~ {j.max_bound()}")
    #     print(f"{j.max_bound()}", end=', ')

    space = JakaSpace(
        (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
        (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi)
    )
    decomp = JakaDecomp(
        (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        (12, 12, 12)
    )
    decomp.set_cell_free_vol()
    stp = STP(space, decomp)

    s1 = State(*np.deg2rad([0,0,0,30,0,60,100]))
    s2 = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))

    t = time.time()
    res = stp.solve(s1, s2)
    cost = time.time() - t
    print(f"res: {res}")
    print(f"cost: {cost * 1000.0:.2f} ms")


def plot():
    """
    定义了一个jakaspace类给出了，构建函数的参数。
    jakaspace类继承了space类，space类在stp.py中定义
    """
    space = JakaSpace(
        (-4.963716393, -1.171988593, -2.412917691, -1.171988593, -4.963716393, -4.963716393), #这里参数是关节空间的上下界,对应jaka_single_srv.launch中机械臂的上下限
        (4.963716393, 3.653846789, 2.412917691, 3.653846789, 4.963716393, 4.963716393)
    )
    decomp = JakaDecomp(
        (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        (12, 12, 12)
    )

    stp = STP(space, decomp)
    s1 = State(*np.deg2rad([-30, 60, -45, -17, -115, 0]))
    s2 = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))
    stp.plot2(s1, s2)
    # stp.plot3()

def plot2():


    space = JakaSpace(
        (-4.963716393, -1.171988593, -2.412917691, -1.171988593, -4.963716393, -4.963716393),
        (4.963716393, 3.653846789, 2.412917691, 3.653846789, 4.963716393, 4.963716393)
    )
    decomp = JakaDecomp(
        # (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        # (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        # (12, 12, 12)
        # ycr
        (-0.3, -0.3, 0.0, -np.pi, -np.pi, -np.pi),
        (0.9, 0.66, 0.96, np.pi, np.pi, np.pi),
        (10, 8, 8)
        # (1, 1, 1)
    )
    stp = STP(space, decomp)
    s1 = State(-3.142032476769, 1.0469305156436, -2.0944125557880002, -0.5238789011974, 0.0, 0.0)
    s2 = State(*np.deg2rad([45, 60, -64, 20, -30, 0]))
    # stp.plot2(s1, s2)
    stp.plot3()

def task():
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
    space = JakaSpace(
        (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
        (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi)
    )
    decomp = JakaDecomp(
        (0.397-0.04,0.0833-0.04,0.583-0.04,-np.pi, -np.pi, -np.pi),
        # (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        # (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        (0.397+0.04,0.0833+0.04,0.583+0.04,np.pi, np.pi, np.pi),
        # (12, 12, 12)
        (1,1,1)
    )

    stp = STP(space, decomp)
    s1 = State(*np.deg2rad([0, 0, 0, 30, 0, 60, 100]))
    s2 = State(*np.deg2rad([ 32.939617 , 27.272095  ,-8.98766   ,43.592518 ,-82.58138 ,  60.732075,145.08644 ]))
    # stp.publish()
    stp.collect_data(s1,s2)
    # stp.test_cell()
    #stp.sample_gmm_in_certain_cell(s1,s2,30,mode="both",finetuning=False)


    # mode = "both"
    # print("下面是{}的实验结果".format(mode))
    # for j in range(1,7):
    #     step_size = j
    #     print("-----------------------------------------------------------------------------------------------")
    #     print("当前为微调步长{}的结果".format(step_size))
    #     for i in range(2,10):
    #         print("--------------------------------------------------------")
    #         print("下面是经验{}%的采样结果".format(i*10))
    #         stp.sample_gmm_in_certain_cell(s1,s2,10*i,mode=mode,finetuning=True,step_size=step_size) 



    # mode = "pos"
    # print("下面是{}的实验结果".format(mode))
    # for i in range(2,10):
    #     print("下面是经验{}%的采样结果".format(i*10))
    #     stp.sample_gmm_in_certain_cell(s1,s2,10*i,mode=mode)
    # mode = "orien"
    # print("下面是{}的实验结果".format(mode))
    # for i in range(2,10):
    #     print("下面是经验{}%的采样结果".format(i*10))
    #     stp.sample_gmm_in_certain_cell(s1,s2,10*i,mode=mode)    

    # time.sleep(10)
    #stp.samAndeval_in_certain_cell(s1, s2)
    #stp.test(s1,s2)
    #stp.samAndeval_in_certain_cell(s1,s2)
    # stp.sample_gmm_in_certain_cell(s1,s2)
    # stp.plot3()
    #State(*np.deg2rad([0, 120, -90, 150, 90, 0])),
    #State(*np.deg2rad([-80, 170, -60, 150, 180, 0]))

def get_ik_results_from_potential():
    points = np.array([[0.02, 0.02, 0.06],
                       [0.07, 0.03, 0.07],
                       [0.07, 0.03, 0.03],
                       [0.05, 0.03, 0.03],
                       [0.07, 0.03, 0.01],
                       [0.05, 0.03, 0.01],
                       [-0.02, -0.02, 0.06],
                       [-0.01, -0.05, 0.07],
                       [-0.03, -0.07, 0.07],
                       [-0.01, -0.07, 0.07],
                       [0.05, -0.01, 0.07],
                       [0.05, -0.01, 0.05],
                       [0.03, -0.01, 0.07],
                       [0.01, -0.01, 0.07],
                       [0.05, 0.03, -0.01],
                       [0.07, 0.03, -0.03],
                       [0.06, 0.02, -0.06],
                       [0.06, -0.02, -0.02],
                       [0.06, -0.02, -0.06],
                       [0.01100646, - 0.04402584, - 0.06603875]])
    space = JakaSpace(
        (-4.963716393, -1.171988593, -2.412917691, -1.171988593, -4.963716393, -4.963716393), #这里参数是关节空间的上下界,对应jaka_single_srv.launch中机械臂的上下限
        (4.963716393, 3.653846789, 2.412917691, 3.653846789, 4.963716393, 4.963716393)
    )
    decomp = JakaDecomp(
        (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        (12, 12, 12)
    )

    stp = STP(space, decomp)
    s1 = State(*np.deg2rad([0, 120, -90, 150, 90, 0]))
    s2 = State(*np.deg2rad([-15, 0, 84, 100,-254, 0]))
    #stp.test(s1, s2)
    stp.compute_6d_ik_potential(points)
if __name__ == '__main__':
    #print("Python executable:", sys.executable)
    rospy.init_node("main")
    # planar()
    #jaka()
    #for i in range(15):
    task()
    #get_ik_results_from_potential()
