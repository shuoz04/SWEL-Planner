#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/4/18 下午12:36
"""
import time
from typing import Union, Tuple, List, Dict

import moveit_commander
import numpy as np
import rospy



import moveit_msgs.srv
import geometry_msgs.msg
import tf.transformations
import pickle

import numpy as np
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
import os

from sklearn.cluster import KMeans
import pickle

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



