#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/4/18 下午12:37
"""
import collections
import heapq
import threading
import os
# from main_cbh import WeightedGMM, exp_map
import moveit_commander
import numpy as np
import time
import networkx as nx
from typing import List, Iterable, Tuple, Any, Union, Generic, TypeVar, Dict, Callable, Sequence
from copy import deepcopy
import pickle
import rospy
import moveit_msgs.msg
import moveit_msgs.srv
import trajectory_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import tf.transformations
import std_msgs.msg
import sensor_msgs.msg
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation as R

T = TypeVar('T')

np.random.seed(2)

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










class Magic:
    DataType = np.double


def wrap_to_pi(angle: float or List or np.ndarray):
    if isinstance(angle, float):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    else:
        return [wrap_to_pi(a) for a in angle]


class UnionFindSet(Generic[T]):
    def __init__(self):
        self._data = {}

    def union(self, x: T, y: T):
        self._data[self.get_root(x)] = self.get_root(y)

    def get_root(self, x: T) -> T:
        tmp = self._data[x]
        if x != tmp:
            self._data[x] = self.get_root(tmp)
        return tmp


class Ratio:
    def __init__(self, initial=0.0):
        self._data = [0, 0]
        self._value = initial
        self._should_update = False

    def increase(self, num=1):
        self._data[1] += num
        self._data[0] += num
        self._should_update = True

    def increase_total(self, num=1):
        self._data[0] += num
        self._should_update = True

    @property
    def value(self) -> float:
        if self._should_update:
            self._value = self._data[1] / self._data[0]
        return self._value


class StateSet:
    def __init__(self):
        self._data = []

    def add(self, s: 'State'):
        self._data.append(s)

    def sample(self) -> Union['State', None]:
        if self.empty():
            return None
        return self._data[np.random.randint(0, len(self._data))]

    def empty(self) -> bool:
        return len(self._data) == 0

    def __iter__(self):
        return self._data.__iter__()


class State:
    def __init__(self, *vals: float):
        assert len(vals) > 0
        self._data = np.array(vals, Magic.DataType)
        self._dim = self._data.shape[-1]
        self._uid = id(self)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def data_view(self) -> np.ndarray:
        return self._data

    def expand(self, to_s: 'State', ratio=1.0) -> 'State':
        return State(*((to_s.data_view - self._data) * ratio + self._data))

    @property
    def uid(self) -> int:
        return self._uid

    def copy(self) -> 'State':
        s = State(*self._data)
        s._uid = self._uid
        return s

    def __eq__(self, other: 'State') -> bool:
        return other.uid == self._uid or np.allclose(self._data, other.data_view)

    def __getitem__(self, idx) -> Magic.DataType:
        return self._data[idx]

    def __hash__(self):
        return hash(self._uid)

    def __str__(self):
        return f'State{str(self._data)}'

    def __repr__(self):
        return self.__str__()


class Space:
    def __init__(self, lb: Tuple, ub: Tuple, check_motion_resolution=0.08):
        assert len(lb) == len(ub)
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._check_motion_resolution = check_motion_resolution

    @property
    def lb(self) -> np.ndarray:
        return self._lb.copy()

    @property
    def ub(self) -> np.ndarray:
        return self._ub.copy()

    @staticmethod
    def distance(s1: State, s2: State) -> float:
        return np.linalg.norm(s1.data_view - s2.data_view)

    def check_validity(self, s: State) -> bool:
        raise NotImplementedError

    def check_motion(self, s1: State, s2: State) -> bool:
        count = np.ceil(self.distance(s1, s2) / self._check_motion_resolution)  # np.float64
        q: List[Tuple[int, int]] = [(1, count)]
        while q:
            i1, i2 = q.pop()
            mid = (i1 + i2) // 2
            if not self.check_validity(s1.expand(s2, mid / count)):
                return False
            if i1 < mid:
                q.append((i1, mid - 1))
            if i2 > mid:
                q.append((mid + 1, i2))
        return True

    def sample_uniform(self) -> State:
        return State(*np.random.uniform(self._lb, self._ub))##返回6维度的随机均匀state


class Cell:
    def __init__(self, rid: Tuple, ws: Space):
        self._rid = rid
        self._dim = len(rid)
        self._neighbors = tuple()
        self._ws = ws
        #
        self._start_set = StateSet()
        #
        self.free_vol = 1.0
        self.total_states = []

    def set_neighbors(self, nbrs: Tuple['Cell']):
        self._neighbors = nbrs

    @property
    def neighbors(self) -> Tuple['Cell']:
        return self._neighbors

    # with same sequence as Cell.neighbors
    @property
    def border_centers(self) -> Sequence[np.ndarray]:                                   ###计算了两个栅格边界的中心点位置
        return [(cell.center_pos + self.center_pos) / 2 for cell in self.neighbors]

    @property
    def rid(self) -> Tuple:
        return self._rid

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def center_pos(self) -> np.ndarray:
        return ((self.ws.ub + self.ws.lb) / 2.0)[:self.dim]

    @property
    def ws(self) -> Space:
        return self._ws

    def __hash__(self):
        return hash(self._rid)

    @property
    def is_connect_to_start(self) -> bool:
        return not self._start_set.empty()

    @property
    def start_set(self) -> StateSet:
        return self._start_set
"""
    decomp = JakaDecomp(
        (-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi),
        (1.0, 1.0, 1.0, np.pi, np.pi, np.pi),
        (12, 12, 12)
    )
"""

class Decomposition:                                                   #
    def __init__(self, lb: Tuple, ub: Tuple, slices: Tuple):
        assert len(lb) == len(ub)
        self._dim = len(slices)
        slices = tuple(slices[i] if i < len(slices) else 1 for i in range(len(lb)))
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._interval = ((self._ub - self._lb) / slices)
        #
        self._cells_dict = {}
        grid_graph: nx.Graph = nx.grid_graph(slices)
        for rid in grid_graph.nodes:
            rid = rid[::-1]
            simple_rid = rid[:self.dim]
            self._cells_dict[simple_rid] = Cell(simple_rid,
                                                Space(self._lb + self._interval * rid,
                                                      self._lb + self._interval * tuple(map(lambda x: x + 1, rid))))
        for rid in grid_graph.nodes:
            self._cells_dict[rid[::-1][:self.dim]].set_neighbors(
                tuple(self._cells_dict[nbr_rid[::-1][:self._dim]] for nbr_rid in grid_graph.neighbors(rid)))
        #
        self._connecty_dict: Dict[Tuple, Ratio] = collections.defaultdict(lambda: Ratio(initial=0.9))
        self.set_cell_free_vol()

    def set_cell_free_vol(self):
        self._set_cell_free_vol(self._cells_dict)

    @staticmethod
    def _set_cell_free_vol(cells_dict: Dict[Tuple, Cell]):
        raise NotImplementedError

    def get_connecty_ratio(self, c1: Cell, c2: Cell) -> Ratio:
        assert c1.rid != c2.rid
        return self._connecty_dict[tuple(sorted([c1.rid, c2.rid]))]

    @property
    def dim(self) -> int:
        return self._dim

    def project(self, s: State) -> Cell:
        ws_s = self.fk(s)
        rid = (ws_s.data_view - self._lb) / self._interval
        rid = tuple(map(int, rid[:self.dim]))
        return self._cells_dict[rid]

    def fk(self, s: State) -> State:
        raise NotImplementedError

    def _sample_in_cell(self, cell: Cell, seed: Union[None, State]) -> State:
        raise NotImplementedError

    def sample_in_cell(self, cell: Cell) -> Union[State, None]:
        nbr_cells: List[Cell] = [cell, *cell.neighbors]
        np.random.shuffle(nbr_cells)
        seed = None
        for _cell in nbr_cells:
            seed = _cell.start_set.sample()           #从cell中的state集合中随机取一个state
            if seed:
                break
        if seed:
            return self._sample_in_cell(cell, seed=seed)
        else:
            return self._sample_in_cell(cell, seed=None)

    def get_cell(self, rid: Tuple) -> Cell:
        return self._cells_dict[rid]

    def get_all_cells(self) -> Sequence[Cell]:
        return [self._cells_dict[key] for key in self._cells_dict]

class STP:
    def __init__(self, space: Space, decomp: Decomposition):
        self.space = space
        self.decomp = decomp
        #
        self.g = nx.Graph()
        #
        self.timeout_ = 10000.0  # s
        #
        self.viz = Viz()

    def compute_lead(self, start_cell: Cell, goal_cell: Cell) -> Sequence[Cell]:
        class CellNode:
            def __init__(self, w_: float, cell_: Cell, cur_pos: np.ndarray, route_: List[Cell]):
                self.cur_w = w_
                self.route = route_
                self.cell = cell_
                self.cur_pos = cur_pos

            def __lt__(self, other: 'CellNode'):
                return self.cur_w < other.cur_w

        q = [CellNode(0.0, start_cell, start_cell.center_pos, [])]
        visited = set()
        while q:
            node = heapq.heappop(q)
            if node.cell in visited:
                continue
            route = node.route + [node.cell]
            visited.add(node.cell)
            if node.cell.rid == goal_cell.rid:
                return route
            for nbr_cell, border_center in zip(node.cell.neighbors, node.cell.border_centers):
                if nbr_cell in visited:
                    continue
                # calc weight
                distance = np.linalg.norm(node.cur_pos - border_center)
                connecty = self.decomp.get_connecty_ratio(nbr_cell, node.cell).value
                free_vol = nbr_cell.free_vol
                nxt_w = distance * np.exp(-10 * connecty) / (1e-3 + free_vol)
                if node.cell.rid == (8, 4) and nbr_cell.rid == (8, 5):
                    print(f"con: {node.cell.rid} -> {nbr_cell.rid} = {connecty:.3f}")
                #
                heapq.heappush(q, CellNode(node.cur_w + nxt_w, nbr_cell, border_center, route))
        raise RuntimeError

    def add_motion(self, s1: State, s2: State):
        self.g.add_edge(s1, s2, w=self.space.distance(s1, s2))
        self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0.5, 0.8, 0.5, 1.0), dim=self.decomp.dim)
        self.viz.wait_for_gui()

    def solve(self, start: State, goal: State) -> Sequence[State]:
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)

        start_cell.start_set.add(start)

        connect_cell_attempts = 5  # TODO: param

        print("begin loop")
        path = []
        t = time.time()
        while len(path) == 0 and self.timeout_ > time.time() - t:
            self.viz.wait_for_gui("wait for current loop")
            print("compute_lead")
            lead = self.compute_lead(start_cell, goal_cell)

            self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.5))
            self.viz.wait_for_gui("wait for sample")

            num = len(lead)
            for i in range(1, num - 1):
                cell: Cell = lead[i]
                prev_cell: Cell = lead[i - 1]
                if not cell.is_connect_to_start:
                    for _ in range(connect_cell_attempts):
                        new_s = self.decomp.sample_in_cell(cell)
                        if not new_s:
                            continue
                        print(f"successfully sampled in f{cell.rid}")
                        cell.total_states.append(new_s)
                        prev_s = prev_cell.start_set.sample()
                        assert prev_s is not None
                        r = self.decomp.get_connecty_ratio(prev_cell, cell)
                        if self.space.check_motion(new_s, prev_s):
                            cell.start_set.add(new_s)
                            self.add_motion(prev_s, new_s)
                            r.increase()
                            print(f"{cell.rid} is connected to start")
                            break
                        else:
                            r.increase_total()
                    if not cell.is_connect_to_start:
                        break
            if lead[-2].is_connect_to_start:
                print(f"try to connect goal cell")
                for s in lead[-2].start_set:
                    if self.space.check_motion(s, goal):
                        self.add_motion(s, goal)
                        path = nx.dijkstra_path(self.g, start, goal, weight='w')
                        print("found")
                        break
        self.viz.wait_for_gui("wait for finish")
        self.viz.publish_trajectory(path)
        return path

    @staticmethod
    def add_obj(scene, name, xyz, size, frame_id):
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = xyz[0]
        pose_msg.pose.position.y = xyz[1]
        pose_msg.pose.position.z = xyz[2]
        pose_msg.pose.orientation.w = 1.0
        scene.add_box(name, pose_msg, size)
        # wait
        obj = scene.get_objects([name])
        r = rospy.Rate(30)
        while len(obj.keys()) == 0:
            obj = scene.get_objects([name])
            r.sleep()
        rospy.loginfo(f"successfully add box: `{name}`")

    def plot(self, start: State, goal: State):
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        scene.remove_world_object("box")
        self.add_obj(scene, 'box', self.decomp.get_cell((8, 6, 8)).center_pos, (0.075, 0.075, 0.075), 'right_base_link')

        self.decomp.get_cell((8, 6, 8)).free_vol = 0.2
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)

        lead = self.compute_lead(start_cell, goal_cell)
        self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.35))

        s0 = State(*np.deg2rad([-9.1008, 59.98469999, -45.0142, -17.02055001, -115.01136001, 0.]))
        self.viz.publish_motion(self.decomp.fk, [start, s0], (0.8, 0.5, 0.5, 1.0), dim=3)
        s0 = State(*np.deg2rad([-6.02928, 66.67599999, -45., -17., -115., 0.]))
        self.viz.publish_motion(self.decomp.fk, [start, s0], (0.8, 0.5, 0.5, 1.0), dim=3)
        s0 = State(*np.deg2rad([-6.02928, 59.32109999, -47.0603, -25.97915001, -136.17072001, 0.]))
        self.viz.publish_motion(self.decomp.fk, [start, s0], (0.8, 0.5, 0.5, 1.0), dim=3)
        self.viz.wait_for_gui("wait for finish")
        #
        lead2 = [self.decomp.get_cell((9, 5, 8)),
                 self.decomp.get_cell((9, 5, 7)),
                 *lead[2:]]
        self.viz.publish_cells(lead2, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("wait for finish")
        #
        s1 = State(*np.deg2rad([-21.15936, 62.27964999, - 85.30025, 29.92914999, - 105.91056001, - 48.40488]))
        self.viz.publish_motion(self.decomp.fk, [start, s1], (0.5, 0.8, 0.5, 1.0), dim=3)
        s2 = State(*np.deg2rad([-6.02928, 57.85564999, -86.7657, 49.03529999, -96.80976001, -43.40488]))
        self.viz.publish_motion(self.decomp.fk, [s1, s2], (0.5, 0.8, 0.5, 1.0), dim=3)
        s3 = State(*np.deg2rad([15.13008, 57.85564999, -86.7657, 49.03529999, -96.80976001, -76.40488]))
        self.viz.publish_motion(self.decomp.fk, [s2, s3], (0.5, 0.8, 0.5, 1.0), dim=3)
        s4 = State(*np.deg2rad([30.26016, 75.52399999, -107.36495001, 51.99384999, -66.5496, -23.40488]))
        self.viz.publish_motion(self.decomp.fk, [s3, s4], (0.5, 0.8, 0.5, 1.0), dim=3)
        self.viz.publish_motion(self.decomp.fk, [s4, goal], (0.5, 0.8, 0.5, 1.0), dim=3)
        self.viz.wait_for_gui("wait for finish")
        #
        lead3 = [*lead2[:3],
                 self.decomp.get_cell((9, 6, 6)),
                 self.decomp.get_cell((9, 7, 6)),
                 self.decomp.get_cell((8, 7, 6)),
                 *lead2[-2:],
                 ]
        self.viz.publish_cells(lead3, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("wait for finish")
        #

    def plot2(self, start: State, goal: State):

        # ycr edit
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        scene.remove_world_object("box")
        self.add_obj(scene, 'box', self.decomp.get_cell((8, 6, 8)).center_pos, (0.075, 0.075, 0.075), 'right_base_link')
        self.decomp.get_cell((8, 6, 8)).free_vol = 0.2
        # ycr edit

        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)
        # self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("calc 0st lead: ok")

        self.viz.publish_state(self.decomp.fk(start))
        self.viz.publish_state(self.decomp.fk(goal))

     
        lead2 = [self.decomp.get_cell((9, 5, 8)),
                 self.decomp.get_cell((9, 5, 7)),
                 *lead[2:]]
        # for cell in lead2:
        #     print(cell.center_pos)
        self.viz.publish_cells(lead2, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("calc 1st lead: ok")

        start_cell.start_set.add(start)
        g = nx.Graph()
        g.add_node(start)
        g.add_node(goal)
        #
        states = []
        for idx, cell in enumerate(lead2):
            tmp = []
            for _ in range(3):
                s = self.decomp.sample_in_cell(cell)
                if s is None:
                    continue
                if idx == 3 and _ == 0:
                    continue
                cell.start_set.add(s)
                tmp.append(s)
                g.add_node(s)
                # print(s)
                print(self.decomp.fk(s))
                self.viz.publish_state(self.decomp.fk(s))
            states.append(tmp)
        states[0].append(start)
        states[-1].append(goal)
        self.viz.wait_for_gui("sample along 1st lead: ok")
        #
        for i in range(1, len(states)):
            ss1, ss2 = states[i - 1], states[i]
            for s1 in ss1:
                for s2 in ss2:
                    validity = self.space.check_motion(s1, s2)
                    self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
                    if validity:
                        g.add_edge(s1, s2, w=self.space.distance(s1, s2))
            # break
        print("has path?", nx.has_path(g, start, goal))
        self.viz.wait_for_gui("check connecty: ok")
        #
        lead3 = [*lead2[:3],
                 self.decomp.get_cell((9, 6, 6)),
                 self.decomp.get_cell((9, 7, 6)),
                 self.decomp.get_cell((8, 7, 6)),
                 *lead2[-2:],
                 ]
        self.viz.publish_cells(lead3, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("calc 2st lead: ok")
        #
        states3 = [states[2]]
        for i in range(3, 6):
            tmp = []
            for _ in range(3):
                s = self.decomp.sample_in_cell(lead3[i])
                if s is None:
                    continue
                lead3[i].start_set.add(s)
                tmp.append(s)
                self.viz.publish_state(self.decomp.fk(s))
            states3.append(tmp)
        self.viz.wait_for_gui("sample along 3st lead: ok")
        states3.append(states[4])
        #
        for i in range(1, len(states3)):
            ss1, ss2 = states3[i - 1], states3[i]
            for s1 in ss1:
                for s2 in ss2:
                    validity = self.space.check_motion(s1, s2)
                    self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
                    if validity:
                        g.add_edge(s1, s2, w=self.space.distance(s1, s2))
            # break
        self.viz.wait_for_gui("check connecty: ok")
        print("has path?", nx.has_path(g, start, goal))
        path = nx.dijkstra_path(g, start, goal, weight='w')
        # print(path)
        for i in range(1, len(path)):
            s1, s2 = path[i - 1], path[i]
            self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0.9, 0.9, 0.2, 1), dim=3, lw=0.01)

    def plot3(self):
        rospy.sleep(1)
        self.viz.publish_cells(self.decomp.get_all_cells(), (0.8, 0.8, 1.0, 0.3))
        self.viz.wait_for_gui("wait")

    # ycr edit
    def plot4(self):
        # gengxin d hanshu

        rospy.sleep(1)
        self.viz.publish_cells(self.decomp.get_cell(), (0.8, 0.8, 1.0, 0.3))
        self.viz.wait_for_gui("wait")

    def publish(self):
        # lead = []
        # for i in range(10,19):
        #      for j in range(5,19):
        #          for k in range(10,19):
        #              item = self.decomp.get_cell((i,j,k))
        #              lead.append(item)
        # self.viz.publish_cells(lead,(0.8,0.8,1.0,0.5))
        lead = []
        item = self.decomp.get_cell((8,6,9))
        lead.append(item)
        self.viz.publish_cells(lead,(0.8,0.8,1.0,0.5))


    def test(self, start: State, goal: State):

        # ycr edit
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        #scene.remove_world_object("box")
        #self.add_obj(scene, 'box', self.decomp.get_cell((8, 6, 8)).center_pos, (0.075, 0.075, 0.075), 'right_base_link')
        #self.decomp.get_cell((8, 6, 8)).free_vol = 0.2
        # ycr edit
        
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)
        # self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.5))
        #self.viz.wait_for_gui("calc 0st lead: ok")

        self.viz.publish_state(self.decomp.fk(start))
        self.viz.publish_state(self.decomp.fk(goal))
        test_cell = self.decomp.get_cell((16,9,16))
        #self.viz.publish_cells([test_cell],(0.8,0.8,1.0,0.5))
        
        
        # lead2 = [
        #          self.decomp.get_cell((14, 10, 17)),
        #          *lead[3:4],
        #          self.decomp.get_cell((15, 10, 16))
        #          ]
        lead2 = lead[1:]
        #self.decomp.get_cell((16, 7, 15))
        self.viz.publish_cells(lead[1:],(0.8,0.8,1.0,0.5))
        time.sleep(2)
        #ead2 = lead
        # for cell in lead2:
        #     print(cell.center_pos)
        #self.viz.publish_cells([start_cell],(0.8,0.8,1.0,0.5))
        # for cell_le in lead:
        #     lead3 = [cell_le]
        #     self.viz.publish_cells(lead3,(0.8,0.8,1.0,0.5)) 
        #     print(cell_le._rid)
            #self.viz.wait_for_gui("calc 1st lead: ok")
        #self.viz.publish_cells([goal_cell],(0.8,0.8,1.0,0.5))
        #self.viz.publish_cells(lead2, (0.8, 0.8, 1.0, 0.5))
        #self.viz.wait_for_gui("calc 1st lead: ok")
        print("完成珊格可视化")
        start_cell.start_set.add(start)
        g = nx.Graph()
        g.add_node(start)
        g.add_node(goal)
        #
        states = []
        for idx, cell in enumerate(lead2):
            tmp = []
            for _ in range(4):
                s = self.decomp.sample_in_cell(cell)
                while(s is None):
                    s = self.decomp.sample_in_cell(cell)
             #  if s is None:
             #      continue
             #   if idx == 3 and _ == 0:
              #      continue
                cell.start_set.add(s)
                tmp.append(s)
                g.add_node(s)
                #print(s)
                #print(self.decomp.fk(s))
                #s_data = self.decomp.fk(s)
                #print(s_data[0],' ',s_data[1],' ',s_data[2])
                self.viz.publish_state(self.decomp.fk(s))
            states.append(tmp)
        states[0].append(start)
        states[-1].append(goal)
        #self.viz.wait_for_gui("sample along 1st lead: ok")
        
        import random
        for i in range(1, len(states)):
            ss1, ss2 = states[i - 1], states[i]
            for s1 in ss1:
                for s2 in ss2:
                    p = random.random()
                    validity = self.space.check_motion(s1, s2)#此处s1,s2是六维的
                    if p<0.5:
                        self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(0.2, 1, 0.2, 1),dim = 3)
                    else:
                        self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
                    if validity:
                        g.add_edge(s1, s2, w=self.space.distance(s1, s2))
                        #s1_data = self.decomp.fk(s1)
                        #for item in s2_data[:3]:
                         #   print(item)
            # break
        time.sleep(2)
        self.viz.publish_cells(lead2[:2],(0.8,0.8,1.0,0.5))
        self.viz.publish_cells(lead2[2:],(1,1,0,0.5))
        for idx, cell in enumerate(lead2[2:]):
            tmp = []
            for _ in range(30):
                s = self.decomp.sample_in_cell(cell)
                while(s is None):
                    s = self.decomp.sample_in_cell(cell)
             #  if s is None:
             #      continue
             #   if idx == 3 and _ == 0:
              #      continue
                # cell.start_set.add(s)
                # tmp.append(s)
                # g.add_node(s)
                #print(s)
                #print(self.decomp.fk(s))
                #s_data = self.decomp.fk(s)
                #print(s_data[0],' ',s_data[1],' ',s_data[2])
                self.viz.publish_state(self.decomp.fk(s))
        print("has path?", nx.has_path(g, start, goal))
        #self.viz.wait_for_gui("check connecty: ok")
     #  if not nx.has_path(g,start,goal):
        print("start check all paths")
        all_paths = list(nx.all_simple_paths(g, start, goal)) 
        print("have checked ok")
        path = nx.dijkstra_path(g, start, goal, weight='w')
        # print(path)
        for i in range(1, len(path)):
            s1, s2 = path[i - 1], path[i]
            self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
        """
        for path in all_paths:
            for i in range(1, len(path)):
                s1, s2 = path[i - 1], path[i]
                self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
                s_data = self.decomp.fk(path[i])
                for item in s_data[:3]:
                    print(item)
        """
        paths = optimal_path(g,start,goal,max_paths=5)
        """
        for i, (path, length) in enumerate(paths, start=1): 
            for i in range(1, len(path)):
                #s1, s2 = path[i - 1], path[i]
                #self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
                s_data = self.decomp.fk(path[i])
                for item in s_data[:3]:
                    print(item)
        """
        """
        with open('data.txt', 'a') as file:  
            for i, (path, length) in enumerate(paths, start=1):  
             for i in range(1, len(path)-1):  
            # 假设 self.decomp.fk(path[i]) 返回一个包含至少三个元素的序列  
                s_data = self.decomp.fk(path[i])  
                for item in s_data[:3]:  
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                 file.write(str(item) + '\n') 
        """
        with open('data.txt', 'a') as file:  
            for i, (path, length) in enumerate(paths, start=1):  
             for i in range(1, len(path)-1):  
            # 假设 self.decomp.fk(path[i]) 返回一个包含至少三个元素的序列  
                #s_data = self.decomp.fk(path[i])  
                s_data = path[i]
                for item in s_data:  
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                 file.write(str(item) + '\n')        
    def samAndeval_in_certain_cell(self, start: State, goal: State):

        # ycr edit
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        #scene.remove_world_object("box")
        #self.add_obj(scene, 'box', self.decomp.get_cell((8, 6, 8)).center_pos, (0.075, 0.075, 0.075), 'right_base_link')
        #self.decomp.get_cell((8, 6, 8)).free_vol = 0.2
        # ycr edit
        
        #start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        #lead = self.compute_lead(start_cell, goal_cell)
        # self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.5))
        #self.viz.wait_for_gui("calc 0st lead: ok")

        #self.viz.publish_state(self.decomp.fk(start))
        #self.viz.publish_state(self.decomp.fk(goal))
      
        """
        lead2 = [self.decomp.get_cell((9, 5, 8)),
                 self.decomp.get_cell((9, 5, 7)),
                 *lead[2:]]
        """
        #lead2 = lead
        #ead2 = lead
        # for cell in lead2:
        #     print(cell.center_pos)

        lead3 = [goal_cell]
        self.viz.publish_cells(lead3,(0.8,0.8,1.0,0.5)) 
        
            #self.viz.wait_for_gui("calc 1st lead: ok")
            
        #self.viz.publish_cells(lead2, (0.8, 0.8, 1.0, 0.5))
        #self.viz.wait_for_gui("calc 1st lead: ok")

        #start_cell.start_set.add(start)
        g = nx.Graph()
        g.add_node(start)
        g.add_node(goal)
        target_vec = np.array([0.40662378,0.35998749,0.83968215])
        states = []
        for idx, cell in enumerate(lead3):
            tmp = []
            while True:
                s = self.decomp.sample_in_cell(cell)
                while(s is None):
                    s = self.decomp.sample_in_cell(cell)
             #  if s is None:
             #      continue
             #   if idx == 3 and _ == 0:
              #      continue
                cell.start_set.add(s)
                tmp.append(s)
                orien_ = self.decomp.fk(s)
                orien = orien_[3:6]
                pos = orien_[:3]
                point_befor_rotate = np.array([-0.01,0.07,0.19])

                rotation = R.from_euler('xyz', orien, degrees=False)
                original_vector = np.array([0, 0, 1])
                rotated_vector = rotation.apply(original_vector)
                a = rotated_vector
                b = target_vec
                magnitude_a = np.linalg.norm(rotated_vector)
                magnitude_b = np.linalg.norm(target_vec)

                dot = np.dot(a, b)
                if dot < 0:
                    a = -a
                    dot = np.dot(a, b)
        # 计算cos(θ)
                cos_theta = dot / (magnitude_a * magnitude_b)

                angle_radians = np.arccos(cos_theta)
                weight = angle_radians / (0.5*np.pi)
                point_after_rotate = rotation.apply(point_befor_rotate)
                end_pos = point_after_rotate+np.array(pos)
                
                dis = np.array([0.6,0,0.65])- np.array(end_pos)
                dis = np.linalg.norm(dis)
                if dis<0.045:
                    g.add_node(s)
                    self.viz.publish_state(self.decomp.fk(s))
                num_of_states = g.number_of_nodes()
                if num_of_states==2000:break
                #print(s)
                #print(self.decomp.fk(s))
                #s_data = self.decomp.fk(s)
                #print(s_data[0],' ',s_data[1],' ',s_data[2])
                
            states.append(tmp)
        #states[0].append(start)
        #states[-1].append(goal)
        #self.viz.wait_for_gui("sample along 1st lead: ok")
        #
        print('现在采样了{}个状态'.format(num_of_states))
        unvalid_state = []
        for item in g.nodes():
            validity_of_state = self.space.check_validity(item)
            if not validity_of_state:
                unvalid_state.append(item)
        g.remove_nodes_from(unvalid_state)
        num_of_validity_node = g.number_of_nodes()
        print('一共有{}个无碰撞状态'.format(num_of_validity_node))
        """
        # 计算可行边的数量
        for node_1 in g.nodes():
            for node_2 in g.nodes():
                if node_1 == node_2:break
                motion_validity =self.space.check_motion(node_1, node_2)
                if motion_validity:
                    g.add_edge(node_1, node_2, w=self.space.distance(node_1, node_2))
        print('图中一共有{}条可行边连线'.format(g.number_of_edges()))
        """
        """
        for i in range(1, len(states)):
            ss1, ss2 = states[i - 1], states[i]
            for s1 in ss1:
                for s2 in ss2:
                    validity = self.space.check_motion(s1, s2)#此处s1,s2是六维的
                    self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
                    if validity:
                        g.add_edge(s1, s2, w=self.space.distance(s1, s2))
                        #s1_data = self.decomp.fk(s1)
                        #for item in s2_data[:3]:
                         #   print(item)
        """
            # break
        #print("has path?", nx.has_path(g, start, goal))
        #self.viz.wait_for_gui("check connecty: ok")
     #  if not nx.has_path(g,start,goal):
        #print("start check all paths")
        #all_paths = list(nx.all_simple_paths(g, start, goal)) 
        #print("have checked ok")
        #path = nx.dijkstra_path(g, start, goal, weight='w')
        # print(path)
        """
        for i in range(1, len(path)):
            s1, s2 = path[i - 1], path[i]
            self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
        
        for path in all_paths:
            for i in range(1, len(path)):
                s1, s2 = path[i - 1], path[i]
                self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
                s_data = self.decomp.fk(path[i])
                for item in s_data[:3]:
                    print(item)
        """
        #paths = optimal_path(g,start,goal,max_paths=5)
        """
        for i, (path, length) in enumerate(paths, start=1): 
            for i in range(1, len(path)):
                #s1, s2 = path[i - 1], path[i]
                #self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
                s_data = self.decomp.fk(path[i])
                for item in s_data[:3]:
                    print(item)
        """
        """
        with open('data.txt', 'a') as file:  
            for i, (path, length) in enumerate(paths, start=1):  
             for i in range(1, len(path)-1):  
            # 假设 self.decomp.fk(path[i]) 返回一个包含至少三个元素的序列  
                s_data = self.decomp.fk(path[i])  
                for item in s_data[:3]:  
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                 file.write(str(item) + '\n') 
        """
        with open('sample_6D_state_data.txt', 'a') as file: 
            for node in g.nodes():
                
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
             file.write(str(node._data) + '\n')    
        with open('sample_3D_state_data.txt', 'a') as file_1: 
            for node in g.nodes():
                s_data = self.decomp.fk(node)
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                s_data =  s_data[:3]  
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                file_1.write(str(s_data) + '\n')
    def test_cell(self):
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)

        lead = []
        # item = self.decomp.get_cell((8,6,9))
        item = self.decomp.get_cell((0,0,0))
        lead.append(item)
        self.viz.publish_cells(lead,(0.8,0.8,1.0,0.5))
    def collect_data(self, start: State, goal: State):

        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)

        lead = []
        # item = self.decomp.get_cell((8,6,9))
        item = self.decomp.get_cell((0,0,0))
        lead.append(item)
        self.viz.publish_cells(lead,(0.8,0.8,1.0,0.5))

        g = nx.Graph()
        g.add_node(start)
        g.add_node(goal)
        states = []
        for idx, cell in enumerate(lead):
            tmp = []
            while True:
                s = self.decomp.sample_in_cell(cell)
                while(s is None):
                    s = self.decomp.sample_in_cell(cell)
             #  if s is None:
             #      continue
             #   if idx == 3 and _ == 0:
              #      continue
                cell.start_set.add(s)
                tmp.append(s)
                orien_ = self.decomp.fk(s)


                
                g.add_node(s)
                self.viz.publish_state(self.decomp.fk(s))
                num_of_states = g.number_of_nodes()
                if num_of_states==4000:break
                #print(s)
                #print(self.decomp.fk(s))
                #s_data = self.decomp.fk(s)
                #print(s_data[0],' ',s_data[1],' ',s_data[2])
                
            states.append(tmp)
        #states[0].append(start)
        #states[-1].append(goal)
        #self.viz.wait_for_gui("sample along 1st lead: ok")
        #
        print('现在采样了{}个状态'.format(num_of_states))
        unvalid_state = []
        for item in g.nodes():
            validity_of_state = self.space.check_validity(item)
            if not validity_of_state:
                unvalid_state.append(item)
        g.remove_nodes_from(unvalid_state)
        num_of_validity_node = g.number_of_nodes()
        print('一共有{}个无碰撞状态'.format(num_of_validity_node))
        j =0
        
        # 计算可行边的数量
        for node_1 in g.nodes():
            for node_2 in g.nodes():
                if node_1 == node_2:break
                motion_validity =self.space.check_motion(node_1, node_2)
                if motion_validity:
                    g.add_edge(node_1, node_2, w=self.space.distance(node_1, node_2))
                j=j+1
                
        print('图中一共有{}条可行边连线'.format(g.number_of_edges()))

        
        """
        for i in range(1, len(states)):
            ss1, ss2 = states[i - 1], states[i]
            for s1 in ss1:
                for s2 in ss2:
                    validity = self.space.check_motion(s1, s2)#此处s1,s2是六维的
                    self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
                    if validity:
                        g.add_edge(s1, s2, w=self.space.distance(s1, s2))
                        #s1_data = self.decomp.fk(s1)
                        #for item in s2_data[:3]:
                         #   print(item)
        """
            # break
        #print("has path?", nx.has_path(g, start, goal))
        #self.viz.wait_for_gui("check connecty: ok")
     #  if not nx.has_path(g,start,goal):
        #print("start check all paths")
        #all_paths = list(nx.all_simple_paths(g, start, goal)) 
        #print("have checked ok")
        #path = nx.dijkstra_path(g, start, goal, weight='w')
        # print(path)
        """
        for i in range(1, len(path)):
            s1, s2 = path[i - 1], path[i]
            self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
        
        for path in all_paths:
            for i in range(1, len(path)):
                s1, s2 = path[i - 1], path[i]
                self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(0, 0, 0, 1), dim=3, lw=0.01)
                s_data = self.decomp.fk(path[i])
                for item in s_data[:3]:
                    print(item)
        """
        #paths = optimal_path(g,start,goal,max_paths=5)


        with open('sample_6D_state_data.txt', 'a') as file: 
            for node in g.nodes():
                
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                file.write(str(node._data) + '\n')    
                sum_edge = 0
        sum_edge = 0

        
        with open('sample_3D_state_data.txt', 'a') as file_1: 
            
            for node in g.nodes():
                s_data = self.decomp.fk(node)
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                s_data =  s_data[:3]
                w = g.degree(node)
                sum_edge+=w 

                w = w/(num_of_validity_node-1)
                s_data = [s_data,w]
                # 将 item 转换为字符串（如果它还不是字符串的话），然后写入文件，并添加一个换行符  
                file_1.write(str(s_data) + '\n')   
                
            print("平均每个节点的边数：", sum_edge/num_of_validity_node)


    def sample_gmm_in_certain_cell(self, start: State, goal: State, ex_num,mode = "both",finetuning = False,step_size=1):

        lead = []
        # item = self.decomp.get_cell((8,6,9))
        item = self.decomp.get_cell((0,0,0))
        lead.append(item)
        # low = [0.49, 0, 0.50]
        # up = [0.7, 0.16, 0.66]
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        # goal_cell = self.decomp.project(goal)
        # print("rid:",goal_cell._rid)
        # lead3 = [goal_cell]
        self.viz.publish_cells(lead,(0.8,0.8,1.0,0.5)) 
        time.sleep(5)
        g = nx.Graph()
        g.add_node(start)
        g.add_node(goal)
        states = []
        points = []
        num_point = 100
        point_now = 0
        for idx, cell in enumerate(lead):
            tmp = []
            while point_now<100:
                s = self.sample_state_with_ws_and_JS(ex_num,mode,finetuning=finetuning,step_size=step_size)
                while(s is None):
                    s = self.sample_state_with_ws_and_JS(ex_num,mode,finetuning=finetuning,step_size=step_size)
                s_3 = self.decomp.fk(s)
                is_in_scene = True
                if is_in_scene:
                    tmp.append(s)
                    g.add_node(s)
                    self.viz.publish_state(self.decomp.fk(s))
                    point_now += 1

            states.append(tmp)
        # 图g中都是有逆解的，且在指定珊格内的点，以六维数据形式存在，即关节角
        print('现在采样了{}个状态'.format(num_point))
        unvalid_state = []
        for item in g.nodes():
            validity_of_state = self.space.check_validity(item)
            if not validity_of_state:
                unvalid_state.append(item)
        g.remove_nodes_from(unvalid_state)
        num_of_validity_node = g.number_of_nodes()
        print('一共有{}个无碰撞状态'.format(num_of_validity_node))
        for node_1 in g.nodes():
            for node_2 in g.nodes():
                if node_1 == node_2:break
                motion_validity =self.space.check_motion(node_1, node_2)
                if motion_validity:
                    g.add_edge(node_1, node_2, w=self.space.distance(node_1, node_2))
        print('图中一共有{}条可行边连线'.format(g.number_of_edges()))
        sum_edge = 0
        for node in g.nodes():
            w = g.degree(node)
            sum_edge+=w 
        print("平均每个节点的边数：", sum_edge/num_of_validity_node)
    def sample_state_with_ws_and_JS(self,ex_num,mode = "both",finetuning = False,step_size =1):
    # 整体采样函数，整合上方工作空间采样和关节空间采样并作逆解返回state对象
        # seed = State(*np.array(sample_6d(),dtype = float))  # 获得关节空间采样结果
        seed = State(*np.random.uniform(    #均匀采样
        (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
        (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi)
            )) 
        pos, orien = sample_with_weightGMM(ex_num=ex_num,finetuing=finetuning,step_size=step_size)
        x_min, x_max = 0.3137,0.4803
        y_min, y_max = 0, 1/6
        z_min, z_max = 0.5, 0.666

        # 生成一个随机点
        random_pos = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ])
        orien = [orien[1],orien[2],orien[3],orien[0]]
        euler = tf.transformations.euler_from_quaternion(orien)
        data_1 = np.array([0,0,0],dtype = float)
        if mode == "pos":
            data_final = np.hstack((pos,data_1))
            pt = State(*data_final)
            pt.data_view[3:] = self.decomp.fk(seed).data_view[3:]
        if mode == "both":
            data_final = np.hstack((pos,data_1))
            pt = State(*data_final)
            pt.data_view[3:] = euler
        if mode == "orien":
            data_final = np.hstack((random_pos,data_1))
            pt = State(*data_final)
            pt.data_view[3:] = euler                       
        # pt.data_view[3:] = self.decomp.fk(seed).data_view[3:]
        return self.decomp._moveit_ik(pt, seed) 
def optimal_path(g,start:State,goal:State,max_paths):   #返回一个字典，字典包含了该路图下所有可行路径中路径最短的前max_paths条
    paths_with_lengths = []
    all_paths = list(nx.all_simple_paths(g, start, goal))  
    path_lengths = []  
    for path in all_paths:  
        # 使用生成器表达式和sum函数计算总权重  
        total_weight = sum(g[u][v]['w'] for u, v in zip(path[:-1], path[1:]))  
        paths_with_lengths.append((path, total_weight))  
    sorted_paths_with_lengths = sorted(paths_with_lengths, key=lambda x: x[1])
    shortest_paths_with_lengths = sorted_paths_with_lengths[:max_paths] 
    return shortest_paths_with_lengths
def sample_in_pos_gmm():
    # 定义三维高斯分布的均值向量  
    mog_means = [
    np.array([ 0.07046908,  1.30692769, -1.18496925]),
    np.array([-0.06943756 , 1.20132714, -0.92336193]),
    np.array([-0.05707147  ,1.04293068, -0.90826982])
    ]   
    mog_covariances = [
    [[ 2.67495390e-05,  5.77148928e-05,  5.05309612e-05],
    [ 5.77148928e-05 , 1.94071205e-04 , 5.29161277e-05],
    [ 5.05309612e-05 , 5.29161277e-05 , 1.57318012e-04]],

    [[ 6.92769352e-03, -3.47953747e-03,  8.55234075e-03],
    [-3.47953747e-03 , 9.07716398e-03 ,-1.30317239e-02],
    [ 8.55234075e-03, -1.30317239e-02 , 4.31910729e-02]],

    [[ 5.43322176e-03, -1.48590041e-03 , 4.76881060e-03],
    [-1.48590041e-03 , 1.84514037e-03 ,-3.60992673e-03],
    [ 4.76881060e-03 ,-3.60992673e-03 , 8.54771512e-03]]
    ]
 
    mog_weights = [0.01023507, 0.89563609, 0.09412884] # 每个分量的权重
 
# 根据权重随机选择一个高斯分量
    chosen_component = np.random.choice(len(mog_means), p=mog_weights)
 
# 从选择的高斯分量中采样一个点
    sampled_point = multivariate_normal.rvs(mean=mog_means[chosen_component], cov=mog_covariances[chosen_component])

    return sampled_point
def is_in_current_scene(low,up,s:State):
    is_in_scene = 1
    x = s._data[0]
    y = s._data[1]
    z = s._data[2]
    if x > up[0] or x < low[0]: is_in_scene = 0
    if y > up[1] or y < low[1]: is_in_scene = 0
    if z > up[2] or z < low[2]: is_in_scene = 0
    return is_in_scene
def sample_in_orien_gmm():
        # 定义三维高斯分布的均值向量  
    mog_means = [
    np.array([1.14554359, 3.76069647, 2.45144533]),  # 第一个分量的均值
    np.array([0.88711158, 3.78716541, 2.25876497]),  # 第二个分量的均值
    np.array([1.5516782 , 3.7547973 ,2.70434327])  # 第三个分量的均值
    ]   
 
    mog_covariances = [
    [[ 4.33300035e-02, -4.99436249e-04,  5.93312265e-03],
    [-4.99436249e-04,  2.79245882e-05, -3.08262154e-04],
    [ 5.93312265e-03, -3.08262154e-04,  3.63925947e-03]],

    [[ 3.65603231e-02, -5.29988207e-04 , 2.88230165e-03],
    [-5.29988207e-04 , 1.12677079e-04 ,-5.81926904e-04],
    [ 2.88230165e-03 ,-5.81926904e-04,  3.05243854e-03]],

    [[ 3.83234076e-02,  5.75920643e-04 , 1.23939256e-02],
    [ 5.75920643e-04 , 2.19415744e-05 , 3.68573523e-04],
    [ 1.23939256e-02 , 3.68573523e-04 , 7.79992995e-03]]      
    ]
 
    mog_weights = [0.3794022, 0.3821996, 0.2383982]  # 每个分量的权重
 
# 根据权重随机选择一个高斯分量
    chosen_component = np.random.choice(len(mog_means), p=mog_weights)
 
# 从选择的高斯分量中采样一个点
    sampled_point = multivariate_normal.rvs(mean=mog_means[chosen_component], cov=mog_covariances[chosen_component])
    return sampled_point
def sample_in_work_space_with_EM():
    mog_means = [
    np.array([0.53115991, 0.09719978, 0.53595373]),  # 第一个分量的均值
    np.array([0.56462763, 0.12902873, 0.57918368]),  # 第二个分量的均值
    np.array([0.52360095, 0.02124487, 0.52451799])  # 第三个分量的均值
    ]   

    mog_covariances = [
    [[ 5.40573326e-04 ,-6.07175424e-05, -1.03586221e-04],
    [-6.07175424e-05,  1.68721713e-03,  1.46484710e-04],
    [-1.03586221e-04,  1.46484710e-04,  5.81219721e-04]],

    [[ 1.67873699e-03,  4.15616332e-04, -1.33519882e-03],
    [ 4.15616332e-04,  7.39383912e-04, -5.19539469e-05],
    [-1.33519882e-03, -5.19539469e-05,  2.11186146e-03]],

    [[ 1.40624904e-05,  1.56990287e-06, -3.00487976e-06],
    [ 1.56990287e-06,  1.48773246e-06, -5.72264670e-07],
    [-3.00487976e-06, -5.72264670e-07,  2.88935451e-06]]      
    ]
 
    mog_weights = [0.40610975, 0.59057089, 0.00331936]  # 每个分量的权重

# 根据权重随机选择一个高斯分量
    chosen_component = np.random.choice(len(mog_means), p=mog_weights)
 
# 从选择的高斯分量中采样一个点
    sampled_point = multivariate_normal.rvs(mean=mog_means[chosen_component], cov=mog_covariances[chosen_component])
    return sampled_point
def sample_in_work_space_with_EM_and_potential():
    mog_means = [
    np.array([0.53115991, 0.09719978, 0.53595373]),  # 第一个分量的均值
    np.array([0.56462763, 0.12902873, 0.57918368]),  # 第二个分量的均值
    np.array([0.52360095, 0.02124487, 0.52451799])  # 第三个分量的均值
    ] 
    f1 = np.array([ 0.01397475, -0.0904236,  -0.00941188])
    f2 = np.array([ 0.33226185, -0.23796437,  0.19280717])
    f3 = np.array([0.17049813, 0.04186867, 0.07350064])
    f1 = f1 / np.linalg.norm(f1)
    f2 = f2 / np.linalg.norm(f2)
    f3 = f3 / np.linalg.norm(f3)
    delt_t = -0.04
    mog_means[0] += f1 * delt_t 
    mog_means[1] += f2 * delt_t
    mog_means[2] += f3 * delt_t
    mog_covariances = [
    [[ 5.40573326e-04 ,-6.07175424e-05, -1.03586221e-04],
    [-6.07175424e-05,  1.68721713e-03,  1.46484710e-04],
    [-1.03586221e-04,  1.46484710e-04,  5.81219721e-04]],

    [[ 1.67873699e-03,  4.15616332e-04, -1.33519882e-03],
    [ 4.15616332e-04,  7.39383912e-04, -5.19539469e-05],
    [-1.33519882e-03, -5.19539469e-05,  2.11186146e-03]],

    [[ 1.40624904e-05,  1.56990287e-06, -3.00487976e-06],
    [ 1.56990287e-06,  1.48773246e-06, -5.72264670e-07],
    [-3.00487976e-06, -5.72264670e-07,  2.88935451e-06]]      
    ]
 
    mog_weights = [0.40610975, 0.59057089, 0.00331936]  # 每个分量的权重

# 根据权重随机选择一个高斯分量
    chosen_component = np.random.choice(len(mog_means), p=mog_weights)
 
# 从选择的高斯分量中采样一个点
    sampled_point = multivariate_normal.rvs(mean=mog_means[chosen_component], cov=mog_covariances[chosen_component])
    return sampled_point
def sample_6d():
    # 拼接两个三关节采样结果得道六维度数据
    pos = np.array(sample_in_pos_gmm())
    orien = np.array(sample_in_orien_gmm())
    point = np.hstack((pos,orien))
    point = np.array(point,dtype = float)
    return point
def sample_with_weightGMM(ex_num,finetuing = False,step_size =1):
    import os
    if finetuing:
        with open("./gmm/pos/pos_finetuning/step_size{}cm/gmm_{}%_pos_finetuning.pkl".format(step_size,ex_num), "rb") as f:
            gmm_pos = pickle.load(f)
    else:
        with open("./gmm/pos/gmm_{}%_pos.pkl".format(ex_num), "rb") as f:
            gmm_pos = pickle.load(f)
    with open("./gmm/orien/gmm_{}%_orien.pkl".format(ex_num), "rb") as f:
        gmm_orien = pickle.load(f)
    pos = gmm_pos.sample(1)[0]
    v_orien = gmm_orien.sample(1)
    q_orien = np.array([exp_map(v, gmm_orien.q_ref) for v in v_orien])[0]
    return pos, q_orien

    
class Viz:
    def __init__(self):
        self.cells_pub = rospy.Publisher("/visualization_decomposition", visualization_msgs.msg.MarkerArray, queue_size=1,
                                         latch=True)
        self.gui_sub = rospy.Subscriber('/rviz_visual_tools_gui', sensor_msgs.msg.Joy, callback=self._joy_callback)
        self.con = threading.Condition()
        self.button_status = []

        self.clear_ids = []
        self.cur_id = 100

        self.trajectory_pub = rospy.Publisher('move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)

        self.config = {
            'base_name': 'base_link',
            'joint_names': [f'joint{i}' for i in range(1, 8)],
            # 'base_name': 'base_link',
            # 'joint_names': [f'joint_{i}' for i in range(1, 4)]
        }

    def _gen_marker_id(self) -> int:
        self.cur_id += 1
        return self.cur_id

    def _joy_callback(self, msg: sensor_msgs.msg.Joy):
        with self.con:
            self.button_status = deepcopy(msg.buttons)
            self.con.notify_all()

    def wait_for_gui(self, txt="wait_for_gui..."):
        rospy.loginfo(txt)
        with self.con:
            while True:
                while len(self.button_status) == 0:
                    self.con.wait()
                if self.button_status[1] == 1:
                    self.button_status = []
                    rospy.loginfo("recv continue cmd")
                    break
                elif self.button_status[4] == 1:
                    rospy.loginfo("recv stop cmd, exec exit(0)")
                    exit(0)
                else:
                    rospy.loginfo("Unknown cmd, try again")
                    self.button_status = []
                    continue

    def clear_all(self):
        msg_array = visualization_msgs.msg.MarkerArray()
        for ns, _id in self.clear_ids:
            msg = visualization_msgs.msg.Marker()
            msg.id = _id
            msg.ns = ns
            msg.action = msg.DELETEALL
            msg_array.markers.append(msg)
        self.cells_pub.publish(msg_array)

    def publish_trajectory(self, states: Sequence[State]):
        msg = moveit_msgs.msg.DisplayTrajectory()
        msg.model_id = "rm_75_description"
        traj_msg = moveit_msgs.msg.RobotTrajectory()
        traj_msg.joint_trajectory.joint_names = self.config['joint_names']
        for idx, s in enumerate(states):
            pt = trajectory_msgs.msg.JointTrajectoryPoint()
            pt.positions = s.data_view.tolist()
            pt.time_from_start = rospy.Time.from_sec(idx)
            traj_msg.joint_trajectory.points.append(pt)
        msg.trajectory.append(traj_msg)
        msg.trajectory_start.joint_state.name = self.config['joint_names']
        msg.trajectory_start.joint_state.position = states[0].data_view.tolist()
        self.trajectory_pub.publish(msg)

    def publish_cells(self, cells: Sequence[Cell], rgba: Tuple):
        ns = "lead"
        interval = cells[0].ws.ub - cells[0].ws.lb
        msg_array = visualization_msgs.msg.MarkerArray()
        msg = visualization_msgs.msg.Marker()
        msg.header.frame_id = self.config['base_name']
        msg.ns = ns
        msg.id = 1  # self._gen_marker_id()
        self.clear_ids.append((ns, msg.id))
        msg.type = msg.CUBE_LIST
        msg.action = msg.ADD
        msg.scale.x = interval[0]
        msg.scale.y = interval[1]
        msg.scale.z = interval[2] if cells[0].dim > 2 else 0.1
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        color = std_msgs.msg.ColorRGBA(*rgba)
        msg.color = color
        msg.lifetime = rospy.Duration.from_sec(0.0)

        for cell in cells:
            c = (cell.ws.lb + cell.ws.ub) / 2.0
            c = c[:cell.dim]
            if cell.dim == 2:
                c = (*c, 0)
            msg.points.append(geometry_msgs.msg.Point(*c))
        msg_array.markers.append(msg)

        msg = visualization_msgs.msg.Marker()
        msg.header.frame_id = self.config['base_name']
        msg.ns = ns + "_txt"
        msg.lifetime = rospy.Duration.from_sec(0.0)
        msg.scale.z = interval[0] / 3  # text size
        msg.pose.orientation.w = 1.0
        msg.action = msg.ADD
        msg.color = std_msgs.msg.ColorRGBA(0.9, 0.7, 0.5, 0.0)
        for idx, cell in enumerate(cells):
            msg.id = 2 + idx
            self.clear_ids.append((ns, msg.id))
            msg.type = msg.TEXT_VIEW_FACING
            msg.text = f'{cell.rid}'
            c = (cell.ws.lb + cell.ws.ub) / 2.0
            msg.pose.position.x = c[0]
            msg.pose.position.y = c[1]
            msg.pose.position.z = 0.1 if cell.dim == 2 else c[2]
            msg_array.markers.append(deepcopy(msg))
        self.cells_pub.publish(msg_array)
    """
                        self.viz.publish_motion(self.decomp.fk, [s1, s2],
                                            rgba=(1, 0.2, 0.2, 1) if not validity else (0.2, 1, 0.2, 1), dim=3)
    """

    def publish_motion(self, fk: Callable[[State], State], states: List[State], rgba: Tuple, dim=2, lw=0.0025):
        assert len(states) > 1
        ns = "path"
        msg_array = visualization_msgs.msg.MarkerArray()
        msg = visualization_msgs.msg.Marker()
        msg.header.frame_id = self.config['base_name']
        msg.ns = ns
        msg.id = self._gen_marker_id()
        self.clear_ids.append((ns, msg.id))
        msg.type = msg.LINE_STRIP
        msg.action = msg.ADD
        msg.scale.x = lw  # line width
        msg.pose.orientation.w = 1.0
        resolution = 0.01
        for i in range(1, len(states)):
            s1 = states[i - 1]
            s2 = states[i]
            num = int(np.ceil(np.linalg.norm(s1.data_view - s2.data_view) / resolution))
            for j in range(num):
                ws_s = fk(s1.expand(s2, j / num))
                if dim == 2:
                    pt = (*ws_s.data_view[:2], 0.3)
                else:
                    pt = ws_s.data_view[:3]
                msg.points.append(geometry_msgs.msg.Point(*pt))
        msg.color = std_msgs.msg.ColorRGBA(*rgba)
        msg_array.markers.append(msg)
        self.cells_pub.publish(msg_array)
    

    def publish_state(self, s: State):
        ns = "state"
        msg_array = visualization_msgs.msg.MarkerArray()
        msg = visualization_msgs.msg.Marker()
        msg.header.frame_id = self.config['base_name']
        msg.ns = ns
        msg.id = self._gen_marker_id()
        self.clear_ids.append((ns, msg.id))
        msg.type = msg.SPHERE
        msg.action = msg.ADD
        msg.scale.x = msg.scale.y = msg.scale.z = 0.02
        msg.pose.position.x = s[0]
        msg.pose.position.y = s[1]
        msg.pose.position.z = s[2]
        msg.pose.orientation.w = 1.0
        msg.color = std_msgs.msg.ColorRGBA(0.55, 0.8, 1.0, 1.0)
        msg_array.markers.append(msg)
        self.cells_pub.publish(msg_array)


if __name__ == '__main__':
    pass
    # uu = UnionFindSet[State]()
    # uu.union(1, 2)

    # s1 = State(1, 2, 3)
    # s2 = s1.copy()
    # print(s1 == s2)
    # print(s.dim)