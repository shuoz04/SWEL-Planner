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

import moveit_commander
import numpy as np
import time
import networkx as nx
from typing import List, Iterable, Tuple, Any, Union, Generic, TypeVar, Dict, Callable, Sequence
from copy import deepcopy

import rospy
import moveit_msgs.msg
import moveit_msgs.srv
import trajectory_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import tf.transformations
import std_msgs.msg
import sensor_msgs.msg

T = TypeVar('T')

np.random.seed(1)


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
    def __init__(self, lb: Tuple, ub: Tuple, check_motion_resolution=0.05):
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
        return State(*np.random.uniform(self._lb, self._ub))


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
    def border_centers(self) -> Sequence[np.ndarray]:
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

    def __str__(self):
        return f'Cell{str(self.rid)}'

    def __repr__(self):
        return self.__str__()


class Decomposition:
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
            seed = _cell.start_set.sample()
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
        self.g = nx.Graph()  # G = (E,V)
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
            print(lead)
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
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)
        self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.5))
        self.viz.wait_for_gui("calc 0st lead: ok")

        self.viz.publish_state(self.decomp.fk(start))
        self.viz.publish_state(self.decomp.fk(goal))

        lead2 = [self.decomp.get_cell((9, 5, 8)),
                 self.decomp.get_cell((9, 5, 7)),
                 *lead[2:]]
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

        # print("has path?", nx.has_path(g, start, goal))
        # path = nx.dijkstra_path(g, start, goal, weight='w')
        # for i in range(1, len(path)):
        #     s1, s2 = path[i - 1], path[i]
        #     self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=(1.0, 0.5, 0.0, 1), dim=3, lw=0.01)

    def plot3(self):
        rospy.sleep(1)
        self.viz.publish_cells(self.decomp.get_all_cells(), (0.8, 0.8, 1.0, 0.3))
        self.viz.wait_for_gui("wait")


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
            'base_name': 'right_base_link',
            'joint_names': [f'right_joint_{i}' for i in range(1, 7)],
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
        msg.model_id = "planar_robot"
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
        msg.color = std_msgs.msg.ColorRGBA(0.9, 0.7, 0.5, 1.0)
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
