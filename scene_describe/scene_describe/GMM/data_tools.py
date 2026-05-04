"""Data loading and PyBullet-based feature extraction for scene GMM training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF_PATH = REPO_ROOT / "rm_description" / "urdf" / "RM75" / "rm_75.urdf"
DEFAULT_TARGET_ORIENTATION = np.array([0.15496272, 0.21455847, 0.0, 0.9643398], dtype=np.float32)
DEFAULT_TARGET_DIRECTION = R.from_quat(DEFAULT_TARGET_ORIENTATION).apply([0.0, 0.0, 1.0])
DEFAULT_EVALUATION_DIRECTION = np.array([0.40662378, 0.35998749, 0.83968], dtype=np.float32)
DEFAULT_TOOL_POINT = np.array([-0.01, 0.07, 0.19], dtype=np.float32)
DEFAULT_TARGET_POSITION = np.array([0.6, 0.0, 0.65], dtype=np.float32)
DEFAULT_END_EFFECTOR_INDEX = 7
DEFAULT_JOINT_COUNT = 7
WEIGHT_LINE_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _create_bullet_client(gui: bool) -> bullet_client.BulletClient:
    connection_mode = p.GUI if gui else p.DIRECT
    client = bullet_client.BulletClient(connection_mode=connection_mode)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.setGravity(0, 0, -9.81)
    return client


def _resolve_urdf_path(urdf_path: Optional[Path]) -> Path:
    resolved_path = Path(DEFAULT_URDF_PATH if urdf_path is None else urdf_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError("RM75 URDF file not found: {0}".format(resolved_path))
    return resolved_path


def _load_robot(client: bullet_client.BulletClient, urdf_path: Optional[Path]) -> int:
    resolved_path = _resolve_urdf_path(urdf_path)
    robot_id = client.loadURDF(
        str(resolved_path),
        useFixedBase=True,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=client.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        flags=client.URDF_USE_SELF_COLLISION,
    )
    client.stepSimulation()
    return robot_id


def _apply_joint_state(
    client: bullet_client.BulletClient,
    robot_id: int,
    joint_values: Sequence[float],
) -> None:
    for joint_index, value in enumerate(joint_values[:DEFAULT_JOINT_COUNT]):
        client.resetJointState(robot_id, joint_index, float(value))
    client.stepSimulation()


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return quaternions / norms


def _align_direction(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.dot(direction, reference) < 0.0:
        return -direction
    return direction


def _angle_between_vectors(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    magnitude = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if magnitude == 0.0:
        return 0.0
    cosine = np.clip(np.dot(vector_a, vector_b) / magnitude, -1.0, 1.0)
    return float(np.arccos(cosine))


def _parse_bracketed_vector(line: str) -> Sequence[float]:
    stripped = line.strip()
    if not stripped:
        return []
    if stripped[0] == "[" and stripped[-1] == "]":
        stripped = stripped[1:-1]
    return np.fromstring(stripped, sep=" ").tolist()


def data_read(file_path: str) -> np.ndarray:
    """Read the legacy paired-line joint sample format into an ``Nx14`` array."""
    combined_rows = []
    with open(file_path, "r") as handle:
        lines = [line for line in handle.readlines() if line.strip()]

    if len(lines) % 2 != 0:
        raise ValueError("Joint sample file must contain an even number of non-empty lines.")

    for index in range(0, len(lines), 2):
        first_row = _parse_bracketed_vector(lines[index])
        second_row = _parse_bracketed_vector(lines[index + 1])
        combined_rows.append(first_row + second_row)
    return np.asarray(combined_rows, dtype=np.float32)


def data_process(
    data: np.ndarray,
    gui: bool = False,
    urdf_path: Optional[Path] = None,
    end_effector_index: int = DEFAULT_END_EFFECTOR_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert joint samples into normalized end-effector quaternions and positions."""
    client = _create_bullet_client(gui)
    robot_id = _load_robot(client, urdf_path)
    orientations = []
    positions = []

    try:
        for joint_values in np.asarray(data, dtype=np.float32):
            _apply_joint_state(client, robot_id, joint_values)
            link_state = client.getLinkState(robot_id, end_effector_index)
            positions.append(link_state[4])
            orientations.append(link_state[5])
    finally:
        client.disconnect()

    orientation_array = _normalize_quaternions(np.asarray(orientations, dtype=np.float32))
    position_array = np.asarray(positions, dtype=np.float32)
    return orientation_array, position_array


def getWeightsforData(
    data: np.ndarray,
    gui: bool = False,
    urdf_path: Optional[Path] = None,
    end_effector_index: int = DEFAULT_END_EFFECTOR_INDEX,
    distance_threshold: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute legacy sample weights from end-effector pose quality."""
    client = _create_bullet_client(gui)
    robot_id = _load_robot(client, urdf_path)
    weights = []
    angles = []

    try:
        for joint_values in np.asarray(data, dtype=np.float32):
            _apply_joint_state(client, robot_id, joint_values)
            link_state = client.getLinkState(robot_id, end_effector_index)

            gripper_orientation = R.from_quat(link_state[5])
            gripper_z_axis = _align_direction(
                gripper_orientation.apply([0.0, 0.0, 1.0]),
                DEFAULT_TARGET_DIRECTION,
            )
            angles.append(_angle_between_vectors(gripper_z_axis, DEFAULT_TARGET_DIRECTION))

            gripper_position = np.asarray(link_state[4], dtype=np.float32)
            end_position = gripper_orientation.apply(DEFAULT_TOOL_POINT) + gripper_position
            distance = np.linalg.norm(DEFAULT_TARGET_POSITION - end_position)
            weight = 0.1 if distance > distance_threshold else distance / distance_threshold
            weights.append(weight)
    finally:
        client.disconnect()

    return np.asarray(weights, dtype=np.float32), np.asarray(angles, dtype=np.float32)


def evaluate_sample_data(orientation_data: np.ndarray) -> np.ndarray:
    """Evaluate sampled quaternions against the reference tool direction."""
    scores = []
    for quaternion in np.asarray(orientation_data, dtype=np.float32):
        rotated_axis = R.from_quat(quaternion).apply([0.0, 0.0, 1.0])
        rotated_axis = _align_direction(rotated_axis, DEFAULT_EVALUATION_DIRECTION)
        angle = _angle_between_vectors(rotated_axis, DEFAULT_EVALUATION_DIRECTION)
        scores.append(angle / (0.5 * np.pi))
    return np.asarray(scores, dtype=np.float32)


def read_weights(path: str) -> np.ndarray:
    """Read the trailing scalar weight stored on each legacy sample line."""
    values = []
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            matches = WEIGHT_LINE_PATTERN.findall(line)
            if matches:
                values.append(float(matches[-1]))
    return np.asarray(values, dtype=np.float32)


__all__ = [
    "DEFAULT_END_EFFECTOR_INDEX",
    "DEFAULT_TARGET_ORIENTATION",
    "DEFAULT_TARGET_POSITION",
    "DEFAULT_URDF_PATH",
    "data_process",
    "data_read",
    "evaluate_sample_data",
    "getWeightsforData",
    "read_weights",
]

