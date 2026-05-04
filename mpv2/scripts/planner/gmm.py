#!/usr/bin/env python3
"""Quaternion utilities and a lightweight weighted GMM implementation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


QuaternionLike = Sequence[float] | np.ndarray


def _normalize_quaternion(quaternion: QuaternionLike) -> np.ndarray:
    normalized = np.asarray(quaternion, dtype=np.float64)
    return normalized / np.linalg.norm(normalized)


def quaternion_multiply(q1: QuaternionLike, q2: QuaternionLike) -> np.ndarray:
    """Multiply two quaternions using the Hamilton product."""
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


def log_map(q: QuaternionLike, q_ref: QuaternionLike) -> np.ndarray:
    """Project a quaternion onto the tangent space of ``q_ref``."""
    quaternion = _normalize_quaternion(q)
    reference = _normalize_quaternion(q_ref)

    q_inv = np.array([reference[0], -reference[1], -reference[2], -reference[3]], dtype=np.float64)
    q_diff = quaternion_multiply(quaternion, q_inv)

    theta = np.arccos(np.clip(q_diff[0], -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)

    return (theta / np.sin(theta)) * q_diff[1:]


def exp_map(v: Sequence[float] | np.ndarray, q_ref: QuaternionLike) -> np.ndarray:
    """Map a tangent-space vector back to quaternion space."""
    tangent_vector = np.asarray(v, dtype=np.float64)
    reference = _normalize_quaternion(q_ref)

    theta = np.linalg.norm(tangent_vector)
    if theta < 1e-6:
        return reference.copy()

    q_exp = np.concatenate([[np.cos(theta)], (np.sin(theta) / theta) * tangent_vector])
    return quaternion_multiply(q_exp, reference)


def weighted_quaternion_mean(
    q_list: Sequence[QuaternionLike],
    weights: Sequence[float] | np.ndarray,
    max_iter: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the weighted mean quaternion with an iterative tangent-space update."""
    q_ref = _normalize_quaternion(q_list[0])
    for _ in range(max_iter):
        v_sum = np.zeros(3, dtype=np.float64)
        for quaternion, weight in zip(q_list, weights):
            v_sum += float(weight) * log_map(quaternion, q_ref)

        if np.linalg.norm(v_sum) < eps:
            break

        q_ref = _normalize_quaternion(quaternion_multiply(exp_map(v_sum, q_ref), q_ref))
    return q_ref


class WeightedGMM:
    """A small weighted Gaussian mixture implementation used by the planner scripts."""

    def __init__(self, n_components: int, shared_cov: bool = False, max_iter: int = 100, tol: float = 1e-4):
        self.K = n_components
        self.shared_cov = shared_cov
        self.max_iter = max_iter
        self.tol = tol
        self.q_ref = []

    def fit(self, V: np.ndarray, weights: np.ndarray) -> None:
        """Fit the model on tangent-space vectors and sample weights."""
        V = np.asarray(V, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        kmeans = KMeans(n_clusters=self.K, n_init=10).fit(V)
        self.pi = np.bincount(kmeans.labels_, weights=weights) / np.sum(weights)
        self.mu = kmeans.cluster_centers_

        if self.shared_cov:
            covariance = np.cov(V.T, aweights=weights)
            self.Sigma = np.stack([covariance] * self.K, axis=0)
        else:
            self.Sigma = np.array(
                [
                    np.cov(V[kmeans.labels_ == k].T, aweights=weights[kmeans.labels_ == k])
                    for k in range(self.K)
                ]
            )

        prev_loglik = -np.inf
        for _ in range(self.max_iter):
            gamma = self._e_step(V, weights)
            self._m_step(V, weights, gamma)

            loglik = self._log_likelihood(V, weights)
            if np.abs(loglik - prev_loglik) < self.tol:
                break
            prev_loglik = loglik

    def _e_step(self, V: np.ndarray, weights: np.ndarray) -> np.ndarray:
        eps = 1e-8
        gamma = np.zeros((len(V), self.K))
        for k in range(self.K):
            diff = V - self.mu[k]
            cov = self.Sigma[k] + 1e-6 * np.eye(3)
            inv_cov = np.linalg.inv(cov)
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            det_cov = np.linalg.det(cov) + eps
            gamma[:, k] = self.pi[k] * np.exp(exp_term) / np.sqrt(det_cov)

        gamma *= weights[:, None]
        gamma_sum = gamma.sum(axis=1, keepdims=True) + eps
        gamma /= gamma_sum
        return gamma

    def _m_step(self, V: np.ndarray, weights: np.ndarray, gamma: np.ndarray) -> None:
        N_k = gamma.sum(axis=0)
        eps = 1e-8
        self.pi = (N_k + eps) / (N_k.sum() + eps)

        for k in range(self.K):
            self.mu[k] = np.sum(gamma[:, k][:, None] * weights[:, None] * V, axis=0) / (
                gamma[:, k] * weights
            ).sum()

        if self.shared_cov:
            cov = np.zeros((3, 3))
            for k in range(self.K):
                diff = V - self.mu[k]
                cov += (
                    gamma[:, k, None, None]
                    * weights[:, None, None]
                    * np.einsum("ni,nj->nij", diff, diff)
                ).sum(axis=0)
            covariance = cov / N_k.sum()
            self.Sigma = np.stack([covariance] * self.K, axis=0)
            return

        for k in range(self.K):
            diff = V - self.mu[k]
            self.Sigma[k] = (
                gamma[:, k, None, None]
                * weights[:, None, None]
                * np.einsum("ni,nj->nij", diff, diff)
            ).sum(axis=0) / (N_k[k] + eps)

    def sample(self, n_samples: int) -> np.ndarray:
        component_indices = np.random.choice(self.K, p=self.pi, size=n_samples)
        samples = []
        for component_index in component_indices:
            samples.append(np.random.multivariate_normal(self.mu[component_index], self.Sigma[component_index]))
        return np.array(samples)

    def _log_likelihood(self, V: np.ndarray, weights: np.ndarray) -> float:
        loglik = 0.0
        for k in range(self.K):
            diff = V - self.mu[k]
            cov = self.Sigma[k] + 1e-6 * np.eye(3)
            inv_cov = np.linalg.inv(cov)
            exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            loglik += np.sum(
                weights * (np.log(self.pi[k]) + exp_term - 0.5 * np.log(np.linalg.det(2 * np.pi * cov)))
            )
        return float(loglik)


__all__ = [
    "WeightedGMM",
    "exp_map",
    "log_map",
    "quaternion_multiply",
    "weighted_quaternion_mean",
]
