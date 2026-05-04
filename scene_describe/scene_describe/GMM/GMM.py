"""Weighted Gaussian mixture training for orientation and position experience data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

try:
    from . import data_tools as tl
except ImportError:  # pragma: no cover - script execution fallback
    import data_tools as tl


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
DEFAULT_ORIENTATION_COMPONENTS = 1
DEFAULT_POSITION_COMPONENTS = 3
DEFAULT_REGULARIZATION = 1e-6


def quaternion_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Multiply two quaternions using the Hamilton product convention."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def log_map(q: Sequence[float], q_ref: Sequence[float]) -> np.ndarray:
    """Project quaternion ``q`` onto the tangent space around ``q_ref``."""
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_ref /= np.linalg.norm(q_ref)

    q_inverse = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]], dtype=np.float64)
    q_delta = quaternion_multiply(q, q_inverse)
    theta = np.arccos(np.clip(q_delta[0], -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return (theta / np.sin(theta)) * q_delta[1:]


def exp_map(v: Sequence[float], q_ref: Sequence[float]) -> np.ndarray:
    """Map a tangent-space vector back onto the quaternion manifold."""
    v = np.asarray(v, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return q_ref.copy()
    q_exp = np.concatenate([[np.cos(theta)], (np.sin(theta) / theta) * v])
    result = quaternion_multiply(q_exp, q_ref)
    return result / np.linalg.norm(result)


def weighted_quaternion_mean(
    q_list: Sequence[Sequence[float]],
    weights: Sequence[float],
    max_iter: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the weighted mean quaternion via iterative tangent-space updates."""
    quaternions = np.asarray(q_list, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    q_ref = quaternions[0].copy()
    q_ref /= np.linalg.norm(q_ref)

    for _ in range(max_iter):
        tangent_sum = np.zeros(3, dtype=np.float64)
        for quaternion, weight in zip(quaternions, sample_weights):
            tangent_sum += weight * log_map(quaternion, q_ref)
        if np.linalg.norm(tangent_sum) < eps:
            break
        q_ref = exp_map(tangent_sum, q_ref)
        q_ref /= np.linalg.norm(q_ref)
    return q_ref


def _weighted_covariance(
    values: np.ndarray,
    mean: np.ndarray,
    sample_weights: np.ndarray,
    regularization: float,
) -> np.ndarray:
    centered = values - mean
    denominator = sample_weights.sum()
    if denominator <= 0.0:
        return np.eye(values.shape[1], dtype=np.float64) * regularization
    covariance = (sample_weights[:, None, None] * np.einsum("ni,nj->nij", centered, centered)).sum(axis=0)
    covariance /= denominator
    covariance += np.eye(values.shape[1], dtype=np.float64) * regularization
    return covariance


def _gaussian_density(values: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    dimension = values.shape[1]
    centered = values - mean
    inverse = np.linalg.inv(covariance)
    exponent = -0.5 * np.sum(centered @ inverse * centered, axis=1)
    determinant = max(np.linalg.det(covariance), np.finfo(np.float64).eps)
    normalizer = np.sqrt(((2.0 * np.pi) ** dimension) * determinant)
    return np.exp(exponent) / normalizer


class WeightedGMM:
    """A lightweight weighted Gaussian mixture model with optional shared covariance."""

    def __init__(
        self,
        n_components: int,
        shared_cov: bool = False,
        max_iter: int = 200,
        tol: float = 1e-4,
        regularization: float = DEFAULT_REGULARIZATION,
        random_state: Optional[int] = None,
    ) -> None:
        self.K = n_components
        self.shared_cov = shared_cov
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.random_state = random_state
        self.q_ref = []
        self.pi = None
        self.mu = None
        self.Sigma = None

    def fit(self, values: np.ndarray, weights: Sequence[float]) -> "WeightedGMM":
        """Fit the mixture model to weighted data."""
        samples = np.asarray(values, dtype=np.float64)
        sample_weights = np.asarray(weights, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError("Training data must be a 2D array.")
        if len(samples) != len(sample_weights):
            raise ValueError("Training data and weights must have the same length.")
        if len(samples) < self.K:
            raise ValueError("Number of samples must be at least the number of mixture components.")

        sample_weights = sample_weights / sample_weights.sum()
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=self.random_state).fit(samples)
        labels = kmeans.labels_
        self.mu = kmeans.cluster_centers_.astype(np.float64)
        self.pi = np.bincount(labels, weights=sample_weights, minlength=self.K).astype(np.float64)
        self.pi /= self.pi.sum()

        global_covariance = _weighted_covariance(samples, np.average(samples, axis=0, weights=sample_weights), sample_weights, self.regularization)
        if self.shared_cov:
            self.Sigma = np.repeat(global_covariance[None, :, :], self.K, axis=0)
        else:
            covariances = []
            for component_index in range(self.K):
                component_mask = labels == component_index
                component_values = samples[component_mask]
                component_weights = sample_weights[component_mask]
                if len(component_values) == 0:
                    covariances.append(global_covariance.copy())
                    continue
                covariance = _weighted_covariance(
                    component_values,
                    self.mu[component_index],
                    component_weights,
                    self.regularization,
                )
                covariances.append(covariance)
            self.Sigma = np.asarray(covariances, dtype=np.float64)

        previous_log_likelihood = -np.inf
        for _ in range(self.max_iter):
            gamma = self._e_step(samples, sample_weights)
            self._m_step(samples, sample_weights, gamma, global_covariance)
            log_likelihood = self._log_likelihood(samples, sample_weights)
            if np.abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood
        return self

    def _e_step(self, values: np.ndarray, sample_weights: np.ndarray) -> np.ndarray:
        gamma = np.zeros((len(values), self.K), dtype=np.float64)
        for component_index in range(self.K):
            density = _gaussian_density(values, self.mu[component_index], self.Sigma[component_index])
            gamma[:, component_index] = self.pi[component_index] * density
        gamma *= sample_weights[:, None]

        denominator = gamma.sum(axis=1, keepdims=True)
        denominator[denominator == 0.0] = np.finfo(np.float64).eps
        return gamma / denominator

    def _m_step(
        self,
        values: np.ndarray,
        sample_weights: np.ndarray,
        gamma: np.ndarray,
        fallback_covariance: np.ndarray,
    ) -> None:
        effective_mass = gamma.sum(axis=0)
        effective_mass[effective_mass == 0.0] = np.finfo(np.float64).eps
        self.pi = effective_mass / effective_mass.sum()

        for component_index in range(self.K):
            weighted_gamma = gamma[:, component_index] * sample_weights
            denominator = weighted_gamma.sum()
            if denominator <= 0.0:
                continue
            self.mu[component_index] = np.sum(weighted_gamma[:, None] * values, axis=0) / denominator

        if self.shared_cov:
            covariance = np.zeros_like(fallback_covariance)
            for component_index in range(self.K):
                weighted_gamma = gamma[:, component_index] * sample_weights
                covariance += _weighted_covariance(
                    values,
                    self.mu[component_index],
                    weighted_gamma,
                    self.regularization,
                )
            covariance /= self.K
            self.Sigma = np.repeat(covariance[None, :, :], self.K, axis=0)
            return

        for component_index in range(self.K):
            weighted_gamma = gamma[:, component_index] * sample_weights
            if weighted_gamma.sum() <= 0.0:
                self.Sigma[component_index] = fallback_covariance.copy()
                continue
            self.Sigma[component_index] = _weighted_covariance(
                values,
                self.mu[component_index],
                weighted_gamma,
                self.regularization,
            )

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample new observations from the fitted mixture."""
        if self.pi is None or self.mu is None or self.Sigma is None:
            raise RuntimeError("The model must be fitted before sampling.")
        component_indices = np.random.choice(self.K, p=self.pi, size=n_samples)
        samples = [
            np.random.multivariate_normal(self.mu[component_index], self.Sigma[component_index])
            for component_index in component_indices
        ]
        return np.asarray(samples, dtype=np.float64)

    def _log_likelihood(self, values: np.ndarray, sample_weights: np.ndarray) -> float:
        weighted_density = np.zeros(len(values), dtype=np.float64)
        for component_index in range(self.K):
            weighted_density += self.pi[component_index] * _gaussian_density(
                values,
                self.mu[component_index],
                self.Sigma[component_index],
            )
        weighted_density = np.clip(weighted_density, np.finfo(np.float64).eps, None)
        return float(np.sum(sample_weights * np.log(weighted_density)))


def get_gmm_model(data: np.ndarray, weight: Sequence[float], dim: int) -> WeightedGMM:
    """Compatibility wrapper that trains a weighted GMM with ``dim`` components."""
    return WeightedGMM(n_components=dim, shared_cov=False).fit(data, weight)


def _build_data_paths(experience_percent: int, data_root: Path) -> Tuple[Path, Path]:
    joint_path = data_root / "sampling-data" / "joint" / "joint_{0}%.txt".format(experience_percent)
    weight_path = data_root / "sampling-data" / "pos" / "pos_{0}%.txt".format(experience_percent)
    return joint_path, weight_path


def train_scene_models(
    experience_percent: int,
    orientation_components: int = DEFAULT_ORIENTATION_COMPONENTS,
    position_components: int = DEFAULT_POSITION_COMPONENTS,
    data_root: Path = DEFAULT_DATA_DIR,
    output_root: Path = DEFAULT_DATA_DIR / "GMM_model",
    gui: bool = False,
) -> Tuple[WeightedGMM, WeightedGMM, np.ndarray]:
    """Train and persist orientation and position GMMs for one experience ratio."""
    joint_path, weight_path = _build_data_paths(experience_percent, data_root)
    joint_samples = tl.data_read(str(joint_path))
    quaternion_samples, position_samples = tl.data_process(joint_samples, gui=gui)
    sample_weights = tl.read_weights(str(weight_path))

    q_ref = weighted_quaternion_mean(quaternion_samples, sample_weights)
    tangent_vectors = np.asarray([log_map(quaternion, q_ref) for quaternion in quaternion_samples], dtype=np.float64)

    orientation_model = WeightedGMM(n_components=orientation_components, shared_cov=False).fit(
        tangent_vectors,
        sample_weights,
    )
    orientation_model.q_ref = q_ref
    position_model = WeightedGMM(n_components=position_components, shared_cov=False).fit(
        position_samples,
        sample_weights,
    )

    orientation_output = output_root / "orien_model" / "gmm_{0}%_orien.pkl".format(experience_percent)
    position_output = output_root / "pos_model" / "gmm_{0}%_pos.pkl".format(experience_percent)
    orientation_output.parent.mkdir(parents=True, exist_ok=True)
    position_output.parent.mkdir(parents=True, exist_ok=True)

    with orientation_output.open("wb") as handle:
        pickle.dump(orientation_model, handle)
    with position_output.open("wb") as handle:
        pickle.dump(position_model, handle)

    orientation_samples = orientation_model.sample(100)
    quaternion_predictions = np.asarray([exp_map(sample, q_ref) for sample in orientation_samples], dtype=np.float32)
    evaluation_scores = tl.evaluate_sample_data(quaternion_predictions)
    return orientation_model, position_model, evaluation_scores


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Train one or more GMM checkpoints from the command line."""
    parser = argparse.ArgumentParser(description="Train weighted scene GMM models.")
    parser.add_argument(
        "experience",
        nargs="*",
        type=int,
        default=[30],
        help="Experience percentages to train, for example: 20 30 40",
    )
    parser.add_argument("--gui", action="store_true", help="Use the PyBullet GUI while extracting features.")
    args = parser.parse_args(argv)

    for experience_percent in args.experience:
        orientation_model, position_model, evaluation_scores = train_scene_models(
            experience_percent=experience_percent,
            gui=args.gui,
        )
        print(
            "trained {0}% scene models: orientation={1}, position={2}, evaluation_mean={3:.4f}".format(
                experience_percent,
                orientation_model.K,
                position_model.K,
                float(np.mean(evaluation_scores)),
            )
        )


__all__ = [
    "WeightedGMM",
    "exp_map",
    "get_gmm_model",
    "log_map",
    "main",
    "quaternion_multiply",
    "train_scene_models",
    "weighted_quaternion_mean",
]


if __name__ == "__main__":
    main()
