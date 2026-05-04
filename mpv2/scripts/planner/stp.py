#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core planner primitives and legacy visualization helpers for MPV2."""

from __future__ import annotations

import collections
import heapq
import logging
import os
import pickle
import random
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import networkx as nx
import numpy as np
import rospy
import sensor_msgs.msg
import std_msgs.msg
import tf.transformations
import trajectory_msgs.msg
import visualization_msgs.msg
from planner.gmm import WeightedGMM, exp_map, log_map, quaternion_multiply, weighted_quaternion_mean
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)
PLANNER_DIR = Path(__file__).resolve().parent
GMM_DIR = PLANNER_DIR / "gmm"
DEFAULT_TIMEOUT_SECONDS = 10000.0
DEFAULT_CONNECT_CELL_ATTEMPTS = 5
DEFAULT_CONNECTIVITY_PRIOR = 0.9
DEFAULT_SOLVE_EDGE_RGBA = (0.5, 0.8, 0.5, 1.0)
DEFAULT_LEAD_RGBA = (0.8, 0.8, 1.0, 0.5)
DEFAULT_VALID_EDGE_RGBA = (0.2, 1.0, 0.2, 1.0)
DEFAULT_INVALID_EDGE_RGBA = (1.0, 0.2, 0.2, 1.0)
DEFAULT_DEMO_BOX_CELL = (8, 6, 8)
DEFAULT_DEMO_BOX_SIZE = (0.075, 0.075, 0.075)
DEFAULT_DEMO_FRAME_ID = "right_base_link"
DEFAULT_JOINT_NAMES = tuple("joint{0}".format(index) for index in range(1, 8))
DEFAULT_PATH_EXPORT_FILE = "data.txt"
DEFAULT_SAMPLE_6D_EXPORT_FILE = "sample_6D_state_data.txt"
DEFAULT_SAMPLE_3D_EXPORT_FILE = "sample_3D_state_data.txt"

np.random.seed(2)
os.environ.setdefault("OMP_NUM_THREADS", "2")


class Magic:
    """Namespace for planner-wide numeric types."""

    DataType = np.double


def wrap_to_pi(angle: Union[float, List[float], np.ndarray]) -> Union[float, List[float]]:
    """Wrap an angle or an angle collection to ``[-pi, pi)``."""
    if np.isscalar(angle):
        value = float(angle)
        return (value + np.pi) % (2 * np.pi) - np.pi
    return [wrap_to_pi(item) for item in angle]


class UnionFindSet(Generic[T]):
    """Minimal union-find used by legacy experiments."""

    def __init__(self) -> None:
        self._data = {}

    def union(self, x: T, y: T) -> None:
        self._data[self.get_root(x)] = self.get_root(y)

    def get_root(self, x: T) -> T:
        if x not in self._data:
            self._data[x] = x
            return x

        parent = self._data[x]
        if x != parent:
            self._data[x] = self.get_root(parent)
        return self._data[x]


class Ratio:
    """Track successful vs total attempts."""

    def __init__(self, initial: float = 0.0) -> None:
        self._data = [0, 0]
        self._value = initial
        self._should_update = False

    def increase(self, num: int = 1) -> None:
        self._data[1] += num
        self._data[0] += num
        self._should_update = True

    def increase_total(self, num: int = 1) -> None:
        self._data[0] += num
        self._should_update = True

    @property
    def value(self) -> float:
        if self._should_update:
            total, success = self._data
            self._value = self._value if total == 0 else success / total
            self._should_update = False
        return self._value


class StateSet:
    """Container that supports uniform random sampling."""

    def __init__(self) -> None:
        self._data = []

    def add(self, s: "State") -> None:
        self._data.append(s)

    def sample(self) -> Optional["State"]:
        if self.empty():
            return None
        return self._data[np.random.randint(0, len(self._data))]

    def empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class State:
    """Mutable planner state with a stable unique id."""

    __slots__ = ("_data", "_dim", "_uid")

    def __init__(self, *vals: float) -> None:
        if not vals:
            raise ValueError("State requires at least one value.")
        self._data = np.array(vals, dtype=Magic.DataType)
        self._dim = self._data.shape[-1]
        self._uid = id(self)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def data_view(self) -> np.ndarray:
        return self._data

    def expand(self, to_s: "State", ratio: float = 1.0) -> "State":
        return State(*((to_s.data_view - self._data) * ratio + self._data))

    @property
    def uid(self) -> int:
        return self._uid

    def copy(self) -> "State":
        copied = State(*self._data)
        copied._uid = self._uid
        return copied

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return other.uid == self._uid or np.allclose(self._data, other.data_view)

    def __getitem__(self, idx: int) -> Magic.DataType:
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def __hash__(self) -> int:
        return hash(self._uid)

    def __str__(self) -> str:
        return "State{0}".format(str(self._data))

    def __repr__(self) -> str:
        return self.__str__()


class Space:
    """Continuous search space with state and motion validation hooks."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...], check_motion_resolution: float = 0.08) -> None:
        if len(lb) != len(ub):
            raise ValueError("Lower and upper bounds must have the same dimension.")
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
        return float(np.linalg.norm(s1.data_view - s2.data_view))

    def check_validity(self, s: State) -> bool:
        raise NotImplementedError

    def check_motion(self, s1: State, s2: State) -> bool:
        distance = self.distance(s1, s2)
        if distance == 0:
            return self.check_validity(s1)

        count = max(1, int(np.ceil(distance / self._check_motion_resolution)))
        queue = [(1, count)]
        while queue:
            i1, i2 = queue.pop()
            mid = (i1 + i2) // 2
            if not self.check_validity(s1.expand(s2, mid / count)):
                return False
            if i1 < mid:
                queue.append((i1, mid - 1))
            if i2 > mid:
                queue.append((mid + 1, i2))
        return True

    def sample_uniform(self) -> State:
        return State(*np.random.uniform(self._lb, self._ub))


class Cell:
    """One workspace cell in the decomposition graph."""

    def __init__(self, rid: Tuple[int, ...], ws: Space) -> None:
        self._rid = rid
        self._dim = len(rid)
        self._neighbors = tuple()
        self._ws = ws
        self._start_set = StateSet()
        self.free_vol = 1.0
        self.total_states = []

    def set_neighbors(self, nbrs: Tuple["Cell", ...]) -> None:
        self._neighbors = nbrs

    @property
    def neighbors(self) -> Tuple["Cell", ...]:
        return self._neighbors

    @property
    def border_centers(self) -> Sequence[np.ndarray]:
        return [(cell.center_pos + self.center_pos) / 2 for cell in self.neighbors]

    @property
    def rid(self) -> Tuple[int, ...]:
        return self._rid

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def center_pos(self) -> np.ndarray:
        return ((self.ws.ub + self.ws.lb) / 2.0)[: self.dim]

    @property
    def ws(self) -> Space:
        return self._ws

    def __hash__(self) -> int:
        return hash(self._rid)

    @property
    def is_connect_to_start(self) -> bool:
        return not self._start_set.empty()

    @property
    def start_set(self) -> StateSet:
        return self._start_set


class Decomposition:
    """Workspace decomposition used by the planner."""

    def __init__(self, lb: Tuple[float, ...], ub: Tuple[float, ...], slices: Tuple[int, ...]) -> None:
        if len(lb) != len(ub):
            raise ValueError("Lower and upper bounds must have the same dimension.")

        self._dim = len(slices)
        normalized_slices = tuple(slices[index] if index < len(slices) else 1 for index in range(len(lb)))
        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._interval = (self._ub - self._lb) / normalized_slices
        self._cells_dict = {}

        grid_graph = nx.grid_graph(normalized_slices)
        for rid in grid_graph.nodes:
            reversed_rid = rid[::-1]
            simple_rid = reversed_rid[: self._dim]
            lower = self._lb + self._interval * reversed_rid
            upper = self._lb + self._interval * tuple(index + 1 for index in reversed_rid)
            self._cells_dict[simple_rid] = Cell(simple_rid, Space(tuple(lower), tuple(upper)))

        for rid in grid_graph.nodes:
            simple_rid = rid[::-1][: self._dim]
            self._cells_dict[simple_rid].set_neighbors(
                tuple(self._cells_dict[nbr[::-1][: self._dim]] for nbr in grid_graph.neighbors(rid))
            )

        self._connecty_dict = collections.defaultdict(lambda: Ratio(initial=DEFAULT_CONNECTIVITY_PRIOR))
        self.set_cell_free_vol()

    def set_cell_free_vol(self) -> None:
        self._set_cell_free_vol(self._cells_dict)

    @staticmethod
    def _set_cell_free_vol(cells_dict: Dict[Tuple[int, ...], Cell]) -> None:
        raise NotImplementedError

    def get_connecty_ratio(self, c1: Cell, c2: Cell) -> Ratio:
        if c1.rid == c2.rid:
            raise ValueError("Connectivity ratio is only defined for two distinct cells.")
        return self._connecty_dict[tuple(sorted([c1.rid, c2.rid]))]

    @property
    def dim(self) -> int:
        return self._dim

    def project(self, s: State) -> Cell:
        ws_state = self.fk(s)
        rid = (ws_state.data_view - self._lb) / self._interval
        rid = tuple(map(int, rid[: self.dim]))
        return self._cells_dict[rid]

    def fk(self, s: State) -> State:
        raise NotImplementedError

    def _sample_in_cell(self, cell: Cell, seed: Optional[State]) -> Optional[State]:
        raise NotImplementedError

    def sample_in_cell(self, cell: Cell) -> Optional[State]:
        neighbors = [cell]
        neighbors.extend(cell.neighbors)
        np.random.shuffle(neighbors)
        seed = None
        for neighbor in neighbors:
            seed = neighbor.start_set.sample()
            if seed is not None:
                break
        return self._sample_in_cell(cell, seed)

    def moveit_ik(self, workspace_state: State, seed: State) -> Optional[State]:
        """Compatibility wrapper for decomposition IK implementations."""
        ik_solver = getattr(self, "moveit_ik_impl", None)
        if callable(ik_solver):
            return ik_solver(workspace_state, seed)

        legacy_solver = getattr(self, "_moveit_ik", None)
        if callable(legacy_solver):
            return legacy_solver(workspace_state, seed)

        raise NotImplementedError("Decomposition subclasses must provide a MoveIt IK implementation.")

    def get_cell(self, rid: Tuple[int, ...]) -> Cell:
        return self._cells_dict[rid]

    def get_all_cells(self) -> Sequence[Cell]:
        return [self._cells_dict[key] for key in self._cells_dict]


@dataclass(order=True)
class _LeadQueueNode:
    """Priority queue node for the coarse cell search."""

    cur_w: float
    cell: Cell = field(compare=False)
    cur_pos: np.ndarray = field(compare=False)
    route: List[Cell] = field(compare=False, default_factory=list)


class STP:
    """Sampling Task Planner."""

    def __init__(
        self,
        space: Space,
        decomp: Decomposition,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connect_cell_attempts: int = DEFAULT_CONNECT_CELL_ATTEMPTS,
        interactive: bool = True,
        viz: Optional["Viz"] = None,
    ) -> None:
        self.space = space
        self.decomp = decomp
        self.g = nx.Graph()
        self.timeout_ = timeout
        self.connect_cell_attempts = connect_cell_attempts
        self.interactive = interactive
        self.viz = viz if viz is not None else Viz()

    def _wait_for_gui(self, message: str) -> None:
        if self.interactive:
            self.viz.wait_for_gui(message)

    def compute_lead(self, start_cell: Cell, goal_cell: Cell) -> Sequence[Cell]:
        queue = [_LeadQueueNode(0.0, start_cell, start_cell.center_pos, [])]
        visited = set()

        while queue:
            node = heapq.heappop(queue)
            if node.cell in visited:
                continue

            route = node.route + [node.cell]
            visited.add(node.cell)
            if node.cell.rid == goal_cell.rid:
                return route

            for neighbor, border_center in zip(node.cell.neighbors, node.cell.border_centers):
                if neighbor in visited:
                    continue
                distance = np.linalg.norm(node.cur_pos - border_center)
                connectivity = self.decomp.get_connecty_ratio(neighbor, node.cell).value
                next_weight = distance * np.exp(-10 * connectivity) / (1e-3 + neighbor.free_vol)
                heapq.heappush(queue, _LeadQueueNode(node.cur_w + next_weight, neighbor, border_center, route))

        raise RuntimeError("Failed to compute a lead between the start and goal cells.")

    def add_motion(self, s1: State, s2: State) -> None:
        self.g.add_edge(s1, s2, w=self.space.distance(s1, s2))
        self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=DEFAULT_SOLVE_EDGE_RGBA, dim=self.decomp.dim)
        self._wait_for_gui("wait for motion review")

    def _connect_cell_from_previous(self, cell: Cell, prev_cell: Cell) -> bool:
        if cell.is_connect_to_start:
            return True

        ratio = self.decomp.get_connecty_ratio(prev_cell, cell)
        for _ in range(self.connect_cell_attempts):
            new_state = self.decomp.sample_in_cell(cell)
            if new_state is None:
                continue
            cell.total_states.append(new_state)
            previous_state = prev_cell.start_set.sample()
            if previous_state is None:
                break

            if self.space.check_motion(new_state, previous_state):
                cell.start_set.add(new_state)
                self.add_motion(previous_state, new_state)
                ratio.increase()
                LOGGER.info("Cell %s connected to the start frontier.", cell.rid)
                return True

            ratio.increase_total()

        return False

    def solve(self, start: State, goal: State) -> Sequence[State]:
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        start_cell.start_set.add(start)
        self.g.add_node(start)
        self.g.add_node(goal)

        LOGGER.info("Starting STP solve loop.")
        path = []
        if start_cell.rid == goal_cell.rid:
            LOGGER.info("Start and goal project to the same cell %s.", start_cell.rid)
            if self.space.check_motion(start, goal):
                self.add_motion(start, goal)
                path = [start, goal]
                self._wait_for_gui("wait for finish")
                self.viz.publish_trajectory(path)
                return path
            LOGGER.info("Direct same-cell motion is invalid; falling back to the generic solve loop.")

        started_at = time.time()
        while not path and (time.time() - started_at) < self.timeout_:
            self._wait_for_gui("wait for current loop")
            lead = self.compute_lead(start_cell, goal_cell)
            self.viz.publish_cells(lead, DEFAULT_LEAD_RGBA)
            self._wait_for_gui("wait for sample")

            for index in range(1, len(lead) - 1):
                if not self._connect_cell_from_previous(lead[index], lead[index - 1]):
                    break

            if lead[-2].is_connect_to_start:
                LOGGER.info("Attempting to connect the goal cell.")
                for state in lead[-2].start_set:
                    if self.space.check_motion(state, goal):
                        self.add_motion(state, goal)
                        path = nx.dijkstra_path(self.g, start, goal, weight="w")
                        LOGGER.info("Found a path with %d states.", len(path))
                        break

        self._wait_for_gui("wait for finish")
        if path:
            self.viz.publish_trajectory(path)
        else:
            LOGGER.warning("No path found before timeout.")
        return path

    @staticmethod
    def add_obj(
        scene: moveit_commander.PlanningSceneInterface,
        name: str,
        xyz: Sequence[float],
        size: Sequence[float],
        frame_id: str,
    ) -> None:
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = xyz[0]
        pose_msg.pose.position.y = xyz[1]
        pose_msg.pose.position.z = xyz[2]
        pose_msg.pose.orientation.w = 1.0
        scene.add_box(name, pose_msg, size)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if scene.get_objects([name]):
                break
            rate.sleep()
        rospy.loginfo("Successfully added box: `%s`", name)

    def _create_scene_interface(self) -> moveit_commander.PlanningSceneInterface:
        scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(0.5)
        return scene

    def _prepare_demo_scene(
        self,
        box_cell: Tuple[int, int, int] = DEFAULT_DEMO_BOX_CELL,
        box_size: Tuple[float, float, float] = DEFAULT_DEMO_BOX_SIZE,
        frame_id: str = DEFAULT_DEMO_FRAME_ID,
        remove_existing: bool = True,
    ) -> moveit_commander.PlanningSceneInterface:
        scene = self._create_scene_interface()
        if remove_existing:
            scene.remove_world_object("box")
        self.add_obj(scene, "box", self.decomp.get_cell(box_cell).center_pos, box_size, frame_id)
        self.decomp.get_cell(box_cell).free_vol = 0.2
        return scene

    def _create_graph_with_terminals(self, start: State, goal: State) -> nx.Graph:
        graph = nx.Graph()
        graph.add_node(start)
        graph.add_node(goal)
        return graph

    def _sample_state(self, cell: Cell) -> State:
        while True:
            state = self.decomp.sample_in_cell(cell)
            if state is not None:
                return state

    def _sample_state_layers(
        self,
        cells: Sequence[Cell],
        samples_per_cell: int,
        graph: Optional[nx.Graph] = None,
        publish_states: bool = True,
        add_to_start_set: bool = True,
        skip: Optional[Callable[[int, int, State], bool]] = None,
    ) -> List[List[State]]:
        layers = []
        for cell_index, cell in enumerate(cells):
            layer = []
            for sample_index in range(samples_per_cell):
                state = self._sample_state(cell)
                if skip is not None and skip(cell_index, sample_index, state):
                    continue
                if add_to_start_set:
                    cell.start_set.add(state)
                layer.append(state)
                if graph is not None:
                    graph.add_node(state)
                if publish_states:
                    self.viz.publish_state(self.decomp.fk(state))
            layers.append(layer)
        return layers

    def _connect_layers(
        self,
        graph: nx.Graph,
        state_layers: Sequence[Sequence[State]],
        dim: int = 3,
        randomize_valid_visuals: bool = False,
    ) -> int:
        added_edges = 0
        for index in range(1, len(state_layers)):
            prev_states = state_layers[index - 1]
            current_states = state_layers[index]
            for s1 in prev_states:
                for s2 in current_states:
                    is_valid = self.space.check_motion(s1, s2)
                    rgba = DEFAULT_VALID_EDGE_RGBA
                    if not is_valid:
                        rgba = DEFAULT_INVALID_EDGE_RGBA
                    elif randomize_valid_visuals and random.random() >= 0.5:
                        rgba = DEFAULT_VALID_EDGE_RGBA
                    self.viz.publish_motion(self.decomp.fk, [s1, s2], rgba=rgba, dim=dim)
                    if is_valid:
                        graph.add_edge(s1, s2, w=self.space.distance(s1, s2))
                        added_edges += 1
        return added_edges

    def _filter_invalid_nodes(self, graph: nx.Graph) -> int:
        invalid_nodes = [node for node in graph.nodes() if not self.space.check_validity(node)]
        graph.remove_nodes_from(invalid_nodes)
        return len(invalid_nodes)

    def _fully_connect_graph(self, graph: nx.Graph) -> int:
        nodes = list(graph.nodes())
        added_edges = 0
        for index, node_1 in enumerate(nodes):
            for node_2 in nodes[:index]:
                if self.space.check_motion(node_1, node_2):
                    graph.add_edge(node_1, node_2, w=self.space.distance(node_1, node_2))
                    added_edges += 1
        return added_edges

    def _append_joint_states(self, path: Union[str, Path], states: Iterable[State]) -> None:
        with Path(path).open("a") as handle:
            for state in states:
                handle.write("{0}\n".format(str(state.data_view)))

    def _append_workspace_samples(
        self,
        path: Union[str, Path],
        graph: nx.Graph,
        include_normalized_degree: bool = False,
    ) -> None:
        file_path = Path(path)
        node_count = graph.number_of_nodes()
        with file_path.open("a") as handle:
            for node in graph.nodes():
                workspace_state = self.decomp.fk(node).data_view[:3]
                if include_normalized_degree and node_count > 1:
                    degree = graph.degree(node) / float(node_count - 1)
                    handle.write("{0}\n".format(str([workspace_state, degree])))
                else:
                    handle.write("{0}\n".format(str(workspace_state)))

    def _append_path_states(
        self,
        path: Union[str, Path],
        paths_with_lengths: Sequence[Tuple[Sequence[State], float]],
    ) -> None:
        with Path(path).open("a") as handle:
            for route, _ in paths_with_lengths:
                for state in route[1:-1]:
                    for value in state:
                        handle.write("{0}\n".format(value))

    @staticmethod
    def _log_graph_status(graph: nx.Graph, prefix: str) -> None:
        LOGGER.info("%s %d nodes and %d edges.", prefix, graph.number_of_nodes(), graph.number_of_edges())

    def _sample_graph_from_cell(
        self,
        cell: Cell,
        graph: nx.Graph,
        limit: int,
        accept_state: Optional[Callable[[State, State], bool]] = None,
    ) -> None:
        while graph.number_of_nodes() < limit:
            state = self._sample_state(cell)
            cell.start_set.add(state)
            workspace_state = self.decomp.fk(state)
            if accept_state is not None and not accept_state(state, workspace_state):
                continue
            graph.add_node(state)
            self.viz.publish_state(workspace_state)

    def plot(self, start: State, goal: State) -> None:
        self._prepare_demo_scene()
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)
        self.viz.publish_cells(lead, (0.8, 0.8, 1.0, 0.35))

        initial_candidates = [
            State(*np.deg2rad([-9.1008, 59.98469999, -45.0142, -17.02055001, -115.01136001, 0.0])),
            State(*np.deg2rad([-6.02928, 66.67599999, -45.0, -17.0, -115.0, 0.0])),
            State(*np.deg2rad([-6.02928, 59.32109999, -47.0603, -25.97915001, -136.17072001, 0.0])),
        ]
        for candidate in initial_candidates:
            self.viz.publish_motion(self.decomp.fk, [start, candidate], (0.8, 0.5, 0.5, 1.0), dim=3)
        self._wait_for_gui("wait for finish")

        lead2 = [self.decomp.get_cell((9, 5, 8)), self.decomp.get_cell((9, 5, 7))]
        lead2.extend(lead[2:])
        self.viz.publish_cells(lead2, DEFAULT_LEAD_RGBA)
        self._wait_for_gui("wait for finish")

        scripted_path = [
            State(*np.deg2rad([-21.15936, 62.27964999, -85.30025, 29.92914999, -105.91056001, -48.40488])),
            State(*np.deg2rad([-6.02928, 57.85564999, -86.7657, 49.03529999, -96.80976001, -43.40488])),
            State(*np.deg2rad([15.13008, 57.85564999, -86.7657, 49.03529999, -96.80976001, -76.40488])),
            State(*np.deg2rad([30.26016, 75.52399999, -107.36495001, 51.99384999, -66.5496, -23.40488])),
        ]
        previous_state = start
        for state in scripted_path:
            self.viz.publish_motion(self.decomp.fk, [previous_state, state], (0.5, 0.8, 0.5, 1.0), dim=3)
            previous_state = state
        self.viz.publish_motion(self.decomp.fk, [previous_state, goal], (0.5, 0.8, 0.5, 1.0), dim=3)
        self._wait_for_gui("wait for finish")

        lead3 = [lead2[0], lead2[1], lead2[2], self.decomp.get_cell((9, 6, 6)), self.decomp.get_cell((9, 7, 6)), self.decomp.get_cell((8, 7, 6))]
        lead3.extend(lead2[-2:])
        self.viz.publish_cells(lead3, DEFAULT_LEAD_RGBA)
        self._wait_for_gui("wait for finish")

    def plot2(self, start: State, goal: State) -> None:
        self._prepare_demo_scene()
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)
        self._wait_for_gui("calc 0st lead: ok")

        self.viz.publish_state(self.decomp.fk(start))
        self.viz.publish_state(self.decomp.fk(goal))
        lead2 = [self.decomp.get_cell((9, 5, 8)), self.decomp.get_cell((9, 5, 7))]
        lead2.extend(lead[2:])
        self.viz.publish_cells(lead2, DEFAULT_LEAD_RGBA)
        self._wait_for_gui("calc 1st lead: ok")

        start_cell.start_set.add(start)
        graph = self._create_graph_with_terminals(start, goal)
        states = self._sample_state_layers(
            lead2,
            3,
            graph=graph,
            skip=lambda cell_index, sample_index, _state: cell_index == 3 and sample_index == 0,
        )
        states[0].append(start)
        states[-1].append(goal)
        self._wait_for_gui("sample along 1st lead: ok")

        self._connect_layers(graph, states, dim=3)
        LOGGER.info("First lead graph has path: %s", nx.has_path(graph, start, goal))
        self._wait_for_gui("check connecty: ok")

        lead3 = [lead2[0], lead2[1], lead2[2], self.decomp.get_cell((9, 6, 6)), self.decomp.get_cell((9, 7, 6)), self.decomp.get_cell((8, 7, 6))]
        lead3.extend(lead2[-2:])
        self.viz.publish_cells(lead3, DEFAULT_LEAD_RGBA)
        self._wait_for_gui("calc 2st lead: ok")

        states3 = [states[2]]
        states3.extend(self._sample_state_layers(lead3[3:6], 3, graph=None))
        states3.append(states[4])
        self._wait_for_gui("sample along 3st lead: ok")

        self._connect_layers(graph, states3, dim=3)
        self._wait_for_gui("check connecty: ok")
        LOGGER.info("Second lead graph has path: %s", nx.has_path(graph, start, goal))
        path = nx.dijkstra_path(graph, start, goal, weight="w")
        for index in range(1, len(path)):
            self.viz.publish_motion(self.decomp.fk, [path[index - 1], path[index]], rgba=(0.9, 0.9, 0.2, 1.0), dim=3, lw=0.01)

    def plot3(self) -> None:
        rospy.sleep(1)
        self.viz.publish_cells(self.decomp.get_all_cells(), (0.8, 0.8, 1.0, 0.3))
        self._wait_for_gui("wait")

    def plot4(self) -> None:
        self.plot3()

    def publish(self) -> None:
        self.viz.publish_cells([self.decomp.get_cell((8, 6, 9))], DEFAULT_LEAD_RGBA)

    def test(self, start: State, goal: State) -> None:
        self._create_scene_interface()
        start_cell = self.decomp.project(start)
        goal_cell = self.decomp.project(goal)
        lead = self.compute_lead(start_cell, goal_cell)

        self.viz.publish_state(self.decomp.fk(start))
        self.viz.publish_state(self.decomp.fk(goal))
        lead2 = list(lead[1:])
        self.viz.publish_cells(lead2, DEFAULT_LEAD_RGBA)
        time.sleep(2)
        LOGGER.info("Grid visualization finished.")

        start_cell.start_set.add(start)
        graph = self._create_graph_with_terminals(start, goal)
        states = self._sample_state_layers(lead2, 4, graph=graph)
        states[0].append(start)
        states[-1].append(goal)
        self._connect_layers(graph, states, dim=3, randomize_valid_visuals=True)

        time.sleep(2)
        self.viz.publish_cells(lead2[:2], DEFAULT_LEAD_RGBA)
        self.viz.publish_cells(lead2[2:], (1.0, 1.0, 0.0, 0.5))
        for cell in lead2[2:]:
            for _ in range(30):
                self.viz.publish_state(self.decomp.fk(self._sample_state(cell)))

        LOGGER.info("Test graph has path: %s", nx.has_path(graph, start, goal))
        LOGGER.info("Enumerating all simple paths for diagnostics.")
        _ = list(nx.all_simple_paths(graph, start, goal))
        LOGGER.info("Finished path enumeration.")
        path = nx.dijkstra_path(graph, start, goal, weight="w")
        for index in range(1, len(path)):
            self.viz.publish_motion(self.decomp.fk, [path[index - 1], path[index]], rgba=(0.0, 0.0, 0.0, 1.0), dim=3, lw=0.01)
        self._append_path_states(DEFAULT_PATH_EXPORT_FILE, optimal_path(graph, start, goal, max_paths=5))

    def sample_and_eval_in_certain_cell(self, start: State, goal: State) -> None:
        self._create_scene_interface()
        goal_cell = self.decomp.project(goal)
        self.viz.publish_cells([goal_cell], DEFAULT_LEAD_RGBA)

        graph = self._create_graph_with_terminals(start, goal)
        target_vec = np.array([0.40662378, 0.35998749, 0.83968215])
        point_before_rotate = np.array([-0.01, 0.07, 0.19])
        target_position = np.array([0.6, 0.0, 0.65])

        def accept_state(_state: State, workspace_state: State) -> bool:
            orientation = workspace_state.data_view[3:6]
            position = workspace_state.data_view[:3]
            rotation = R.from_euler("xyz", orientation, degrees=False)
            rotated_vector = rotation.apply(np.array([0.0, 0.0, 1.0]))
            if np.dot(rotated_vector, target_vec) < 0:
                rotated_vector = -rotated_vector
            _ = rotated_vector
            end_position = rotation.apply(point_before_rotate) + np.array(position)
            return np.linalg.norm(target_position - end_position) < 0.045

        self._sample_graph_from_cell(goal_cell, graph, limit=2000, accept_state=accept_state)
        self._log_graph_status(graph, "Sampled graph with")
        self._filter_invalid_nodes(graph)
        self._log_graph_status(graph, "Retained collision-free graph with")
        self._append_joint_states(DEFAULT_SAMPLE_6D_EXPORT_FILE, graph.nodes())
        self._append_workspace_samples(DEFAULT_SAMPLE_3D_EXPORT_FILE, graph)

    def samAndeval_in_certain_cell(self, start: State, goal: State) -> None:
        """Backward-compatible wrapper for the legacy camelCase API."""
        self.sample_and_eval_in_certain_cell(start, goal)

    def test_cell(self) -> None:
        self._create_scene_interface()
        self.viz.publish_cells([self.decomp.get_cell((0, 0, 0))], DEFAULT_LEAD_RGBA)

    def collect_data(self, start: State, goal: State) -> None:
        self._create_scene_interface()
        target_cell = self.decomp.get_cell((0, 0, 0))
        self.viz.publish_cells([target_cell], DEFAULT_LEAD_RGBA)

        graph = self._create_graph_with_terminals(start, goal)
        self._sample_graph_from_cell(target_cell, graph, limit=4000)
        self._log_graph_status(graph, "Sampled graph with")
        self._filter_invalid_nodes(graph)
        self._log_graph_status(graph, "Retained collision-free graph with")

        edge_count = self._fully_connect_graph(graph)
        LOGGER.info("Constructed %d collision-free edges.", edge_count)
        self._append_joint_states(DEFAULT_SAMPLE_6D_EXPORT_FILE, graph.nodes())
        self._append_workspace_samples(DEFAULT_SAMPLE_3D_EXPORT_FILE, graph, include_normalized_degree=True)

        if graph.number_of_nodes() > 0:
            total_degree = sum(graph.degree(node) for node in graph.nodes())
            LOGGER.info("Average degree per node: %s", total_degree / float(graph.number_of_nodes()))

    def sample_gmm_in_certain_cell(
        self,
        start: State,
        goal: State,
        ex_num: int,
        mode: str = "both",
        finetuning: bool = False,
        step_size: int = 1,
    ) -> None:
        target_cell = self.decomp.get_cell((0, 0, 0))
        self._create_scene_interface()
        self.viz.publish_cells([target_cell], DEFAULT_LEAD_RGBA)
        time.sleep(5)

        graph = self._create_graph_with_terminals(start, goal)
        while graph.number_of_nodes() < 100:
            state = self.sample_state_with_ws_and_JS(ex_num, mode, finetuning=finetuning, step_size=step_size)
            if state is None:
                continue
            graph.add_node(state)
            self.viz.publish_state(self.decomp.fk(state))

        self._log_graph_status(graph, "Sampled graph with")
        self._filter_invalid_nodes(graph)
        self._log_graph_status(graph, "Retained collision-free graph with")

        edge_count = self._fully_connect_graph(graph)
        LOGGER.info("Constructed %d collision-free edges.", edge_count)
        if graph.number_of_nodes() > 0:
            total_degree = sum(graph.degree(node) for node in graph.nodes())
            LOGGER.info("Average degree per node: %s", total_degree / float(graph.number_of_nodes()))

    def sample_state_with_ws_and_JS(
        self,
        ex_num: int,
        mode: str = "both",
        finetuning: bool = False,
        step_size: int = 1,
    ) -> Optional[State]:
        seed = State(
            *np.random.uniform(
                (-np.pi, -2.27, -np.pi, -2.3, -np.pi, -2.26, -np.pi),
                (np.pi, 2.27, np.pi, 2.3, np.pi, 2.26, np.pi),
            )
        )

        pos, orientation_quaternion = sample_with_weight_gmm(
            ex_num=ex_num,
            finetuning=finetuning,
            step_size=step_size,
        )
        random_pos = np.array(
            [
                np.random.uniform(0.3137, 0.4803),
                np.random.uniform(0.0, 1.0 / 6.0),
                np.random.uniform(0.5, 0.666),
            ]
        )
        quaternion_xyzw = [
            orientation_quaternion[1],
            orientation_quaternion[2],
            orientation_quaternion[3],
            orientation_quaternion[0],
        ]
        euler = tf.transformations.euler_from_quaternion(quaternion_xyzw)
        zeros = np.array([0.0, 0.0, 0.0], dtype=float)

        if mode == "pos":
            workspace_state = State(*np.hstack((pos, zeros)))
            workspace_state.data_view[3:] = self.decomp.fk(seed).data_view[3:]
        elif mode == "orien":
            workspace_state = State(*np.hstack((random_pos, zeros)))
            workspace_state.data_view[3:] = euler
        else:
            workspace_state = State(*np.hstack((pos, zeros)))
            workspace_state.data_view[3:] = euler

        return self.decomp.moveit_ik(workspace_state, seed)


def optimal_path(g: nx.Graph, start: State, goal: State, max_paths: int) -> Sequence[Tuple[Sequence[State], float]]:
    """Return the shortest ``max_paths`` simple paths between ``start`` and ``goal``."""
    paths_with_lengths = []
    for path in nx.all_simple_paths(g, start, goal):
        total_weight = sum(g[u][v]["w"] for u, v in zip(path[:-1], path[1:]))
        paths_with_lengths.append((path, total_weight))
    return sorted(paths_with_lengths, key=lambda item: item[1])[:max_paths]



WORKSPACE_GMM_POTENTIAL_DIRECTIONS = (
    np.array([0.01397475, -0.0904236, -0.00941188]),
    np.array([0.33226185, -0.23796437, 0.19280717]),
    np.array([0.17049813, 0.04186867, 0.07350064]),
)


def _sample_gaussian_mixture(spec: Dict[str, Any]) -> np.ndarray:
    """Draw a sample from a Gaussian mixture specification."""
    component_index = np.random.choice(len(spec["means"]), p=spec["weights"])
    return multivariate_normal.rvs(mean=spec["means"][component_index], cov=spec["covariances"][component_index])


def sample_in_pos_gmm() -> np.ndarray:
    """Sample a 3D position from the legacy position GMM."""
    return _sample_gaussian_mixture(POSITION_GMM)


def is_in_current_scene(low: Sequence[float], up: Sequence[float], s: State) -> int:
    """Return 1 when the state's position lies inside the axis-aligned bounds."""
    position = s.data_view[:3]
    if np.any(position > np.asarray(up)) or np.any(position < np.asarray(low)):
        return 0
    return 1


def sample_in_orien_gmm() -> np.ndarray:
    """Sample an orientation expressed in XYZ Euler angles."""
    return _sample_gaussian_mixture(ORIENTATION_GMM)


def sample_in_work_space_with_EM() -> np.ndarray:
    """Sample a workspace position from the EM-estimated GMM."""
    return _sample_gaussian_mixture(WORKSPACE_GMM)


def sample_in_work_space_with_EM_and_potential() -> np.ndarray:
    """Sample a workspace position from the EM GMM with directional bias."""
    adjusted_means = []
    for mean, direction in zip(WORKSPACE_GMM["means"], WORKSPACE_GMM_POTENTIAL_DIRECTIONS):
        adjusted_means.append(mean + direction / np.linalg.norm(direction) * -0.04)
    spec = {
        "means": tuple(adjusted_means),
        "covariances": WORKSPACE_GMM["covariances"],
        "weights": WORKSPACE_GMM["weights"],
    }
    return _sample_gaussian_mixture(spec)


def sample_6d() -> np.ndarray:
    """Sample a 6D workspace state composed of position and orientation."""
    return np.array(np.hstack((sample_in_pos_gmm(), sample_in_orien_gmm())), dtype=float)


def sample_with_weight_gmm(ex_num: int, finetuning: bool = False, step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Sample position and orientation from the pickled weighted GMM models."""
    if finetuning:
        pos_path = GMM_DIR / "pos" / "pos_finetuning" / "step_size{0}cm".format(step_size) / "gmm_{0}%_pos_finetuning.pkl".format(ex_num)
    else:
        pos_path = GMM_DIR / "pos" / "gmm_{0}%_pos.pkl".format(ex_num)
    orien_path = GMM_DIR / "orien" / "gmm_{0}%_orien.pkl".format(ex_num)

    with pos_path.open("rb") as handle:
        gmm_pos = pickle.load(handle)
    with orien_path.open("rb") as handle:
        gmm_orien = pickle.load(handle)

    pos = gmm_pos.sample(1)[0]
    tangent_orientation = gmm_orien.sample(1)
    quaternion_orientation = np.array([exp_map(vector, gmm_orien.q_ref) for vector in tangent_orientation])[0]
    return pos, quaternion_orientation


def sample_with_weightGMM(
    ex_num: int,
    finetuing: bool = False,
    step_size: int = 1,
    finetuning: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper around :func:`sample_with_weight_gmm`."""
    resolved_finetuning = finetuing if finetuning is None else finetuning
    return sample_with_weight_gmm(ex_num=ex_num, finetuning=resolved_finetuning, step_size=step_size)


class Viz:
    """RViz/MoveIt visualization wrapper used by the planner demos."""

    def __init__(self) -> None:
        self.cells_pub = rospy.Publisher("/visualization_decomposition", visualization_msgs.msg.MarkerArray, queue_size=1, latch=True)
        self.gui_sub = rospy.Subscriber("/rviz_visual_tools_gui", sensor_msgs.msg.Joy, callback=self._joy_callback)
        self.con = threading.Condition()
        self.button_status = []
        self.clear_ids = set()
        self.cur_id = 100
        self.trajectory_pub = rospy.Publisher("move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=1)
        self.config = {
            "base_name": "base_link",
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "model_id": "rm_75_description",
        }

    def _gen_marker_id(self) -> int:
        self.cur_id += 1
        return self.cur_id

    def _joy_callback(self, msg: sensor_msgs.msg.Joy) -> None:
        with self.con:
            self.button_status = deepcopy(msg.buttons)
            self.con.notify_all()

    def wait_for_gui(self, txt: str = "wait_for_gui...") -> None:
        rospy.loginfo(txt)
        with self.con:
            while not rospy.is_shutdown():
                while len(self.button_status) == 0 and not rospy.is_shutdown():
                    self.con.wait()
                if rospy.is_shutdown():
                    return
                if len(self.button_status) > 4 and self.button_status[1] == 1:
                    self.button_status = []
                    rospy.loginfo("recv continue cmd")
                    return
                if len(self.button_status) > 4 and self.button_status[4] == 1:
                    rospy.loginfo("recv stop cmd, exiting")
                    raise SystemExit(0)
                rospy.loginfo("Unknown cmd, try again")
                self.button_status = []

    def clear_all(self) -> None:
        msg_array = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.action = visualization_msgs.msg.Marker.DELETEALL
        msg_array.markers.append(marker)
        self.cells_pub.publish(msg_array)
        self.clear_ids.clear()

    def publish_trajectory(self, states: Sequence[State]) -> None:
        if not states:
            return

        msg = moveit_msgs.msg.DisplayTrajectory()
        msg.model_id = self.config["model_id"]
        trajectory = moveit_msgs.msg.RobotTrajectory()
        trajectory.joint_trajectory.joint_names = self.config["joint_names"]
        for idx, state in enumerate(states):
            point = trajectory_msgs.msg.JointTrajectoryPoint()
            point.positions = state.data_view.tolist()
            point.time_from_start = rospy.Time.from_sec(idx)
            trajectory.joint_trajectory.points.append(point)
        msg.trajectory.append(trajectory)
        msg.trajectory_start.joint_state.name = self.config["joint_names"]
        msg.trajectory_start.joint_state.position = states[0].data_view.tolist()
        self.trajectory_pub.publish(msg)

    def publish_cells(self, cells: Sequence[Cell], rgba: Tuple[float, float, float, float]) -> None:
        if not cells:
            return

        interval = cells[0].ws.ub - cells[0].ws.lb
        msg_array = visualization_msgs.msg.MarkerArray()

        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.config["base_name"]
        marker.ns = "lead"
        marker.id = 1
        self.clear_ids.add((marker.ns, marker.id))
        marker.type = visualization_msgs.msg.Marker.CUBE_LIST
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.scale.x = interval[0]
        marker.scale.y = interval[1]
        marker.scale.z = interval[2] if cells[0].dim > 2 else 0.1
        marker.pose.orientation.w = 1.0
        marker.color = std_msgs.msg.ColorRGBA(*rgba)
        marker.lifetime = rospy.Duration.from_sec(0.0)
        for cell in cells:
            center = (cell.ws.lb + cell.ws.ub) / 2.0
            if cell.dim == 2:
                marker.points.append(geometry_msgs.msg.Point(center[0], center[1], 0.0))
            else:
                marker.points.append(geometry_msgs.msg.Point(*center[:3]))
        msg_array.markers.append(marker)

        for idx, cell in enumerate(cells):
            label = visualization_msgs.msg.Marker()
            label.header.frame_id = self.config["base_name"]
            label.ns = "lead_txt"
            label.id = 2 + idx
            self.clear_ids.add((label.ns, label.id))
            label.type = visualization_msgs.msg.Marker.TEXT_VIEW_FACING
            label.action = visualization_msgs.msg.Marker.ADD
            label.scale.z = interval[0] / 3
            label.pose.orientation.w = 1.0
            label.color = std_msgs.msg.ColorRGBA(0.9, 0.7, 0.5, 0.0)
            center = (cell.ws.lb + cell.ws.ub) / 2.0
            label.text = "{0}".format(cell.rid)
            label.pose.position.x = center[0]
            label.pose.position.y = center[1]
            label.pose.position.z = 0.1 if cell.dim == 2 else center[2]
            msg_array.markers.append(label)

        self.cells_pub.publish(msg_array)

    def publish_motion(
        self,
        fk: Callable[[State], State],
        states: List[State],
        rgba: Tuple[float, float, float, float],
        dim: int = 2,
        lw: float = 0.0025,
    ) -> None:
        if len(states) < 2:
            raise ValueError("publish_motion requires at least two states.")

        msg_array = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.config["base_name"]
        marker.ns = "path"
        marker.id = self._gen_marker_id()
        self.clear_ids.add((marker.ns, marker.id))
        marker.type = visualization_msgs.msg.Marker.LINE_STRIP
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.scale.x = lw
        marker.pose.orientation.w = 1.0
        marker.color = std_msgs.msg.ColorRGBA(*rgba)

        resolution = 0.01
        for index in range(1, len(states)):
            start = states[index - 1]
            end = states[index]
            num = max(1, int(np.ceil(np.linalg.norm(start.data_view - end.data_view) / resolution)))
            for step in range(num + 1):
                ws_state = fk(start.expand(end, step / float(num)))
                if dim == 2:
                    marker.points.append(geometry_msgs.msg.Point(ws_state[0], ws_state[1], 0.3))
                else:
                    marker.points.append(geometry_msgs.msg.Point(*ws_state.data_view[:3]))

        msg_array.markers.append(marker)
        self.cells_pub.publish(msg_array)

    def publish_state(self, s: State) -> None:
        msg_array = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = self.config["base_name"]
        marker.ns = "state"
        marker.id = self._gen_marker_id()
        self.clear_ids.add((marker.ns, marker.id))
        marker.type = visualization_msgs.msg.Marker.SPHERE
        marker.action = visualization_msgs.msg.Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.02
        marker.pose.position.x = s[0]
        marker.pose.position.y = s[1]
        marker.pose.position.z = s[2] if s.dim > 2 else 0.0
        marker.pose.orientation.w = 1.0
        marker.color = std_msgs.msg.ColorRGBA(0.55, 0.8, 1.0, 1.0)
        msg_array.markers.append(marker)
        self.cells_pub.publish(msg_array)


__all__ = [
    "Cell",
    "Decomposition",
    "Magic",
    "Ratio",
    "STP",
    "Space",
    "State",
    "StateSet",
    "UnionFindSet",
    "Viz",
    "WeightedGMM",
    "exp_map",
    "is_in_current_scene",
    "log_map",
    "optimal_path",
    "quaternion_multiply",
    "sample_6d",
    "sample_in_orien_gmm",
    "sample_in_pos_gmm",
    "sample_in_work_space_with_EM",
    "sample_in_work_space_with_EM_and_potential",
    "sample_with_weightGMM",
    "sample_with_weight_gmm",
    "weighted_quaternion_mean",
    "wrap_to_pi",
]


if __name__ == "__main__":
    pass
