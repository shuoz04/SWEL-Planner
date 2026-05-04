#!/usr/bin/env python3
"""Backward-compatible exports for the GMM helpers used by legacy scripts."""

from planner.gmm import WeightedGMM, exp_map, log_map, quaternion_multiply, weighted_quaternion_mean

__all__ = [
    "WeightedGMM",
    "exp_map",
    "log_map",
    "quaternion_multiply",
    "weighted_quaternion_mean",
]
