#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhiyu YANG
@email: ZhiyuYANG96@outlook.com
@time: 2022/5/3 下午2:20
"""
import logging
import numpy as np
from numpy import cos, sin, pi, sqrt
from numpy import arccos as acos
from numpy import arcsin as asin
from numpy import arctan2 as atan2
from copy import deepcopy
from typing import List, Tuple


class AnalyticalIK:
    def __init__(self, prefix: str, robot_type='zu7_v2', ):
        """
        d1 a2 a3 d5
                    d6
                    d4
        :param robot_type:
        """
        if robot_type == 'zu7_v1':
            self.d1 = 0.115
            self.d4 = 0.116
            self.d5 = 0.103
            self.d6 = 0.095
            self.a2 = -0.353
            self.a3 = -0.303

        elif robot_type == 'zu7_v2':
            self.d1 = 0.12015
            self.d4 = 0.1135
            self.d5 = 0.1135
            self.d6 = 0.107
            self.a2 = -0.360
            self.a3 = -0.3035

        joint_limits_tools = JointLimitsTool()
        self.bounds = joint_limits_tools.to_array(prefix=prefix)

        # alpha, a, d, theta
        self.dh = [
            [pi / 2, 0, self.d1, 0],
            [0, self.a2, 0, 0],
            [0, self.a3, 0, 0],
            [pi / 2, 0, self.d4, 0],
            [-pi / 2, 0, self.d5, 0],
            [0, 0, self.d6, 0]]

    @staticmethod
    def _real_jvals_to_self(joint_vals):
        ans = deepcopy(joint_vals)
        ans[0] += pi
        ans[1] = -ans[1]
        ans[2] = -ans[2]
        ans[3] = -ans[3]
        ans[4] += pi
        return ans

    def set_joints_with_real_j_vals(self, vals):
        tmp = self._real_jvals_to_self(vals)
        for i in range(6):
            self.dh[i][-1] = tmp[i]

    # 仅测试使用
    def forward_kinematics(self, from_id, to_id) -> Transform:
        """
        joint id from 1 to 6
        forward kinematics, point in frame `from_id` can be transformed to frame `to_id`
        :param from_id: bottom, 1~6
        :param to_id: top, 0~5
        :return:
        """
        t = np.eye(4)
        if from_id == to_id:
            return t
        for i in range(min(from_id, to_id), max(from_id, to_id)):
            alpha = self.dh[i][0]
            a = self.dh[i][1]
            d = self.dh[i][2]
            theta = self.dh[i][3]

            c_theta = cos(theta)
            s_theta = sin(theta)
            c_alpha = cos(alpha)
            s_alpha = sin(alpha)
            t = np.matmul(t, [[c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
                              [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
                              [0, s_alpha, c_alpha, d],
                              [0, 0, 0, 1]])
            # t = np.matmul(t, [[c_theta, -s_theta, 0, a],
            #                   [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * d],
            #                   [s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * d],
            #                   [0, 0, 0, 1]])
        return Transform(t) if from_id > to_id else Transform(np.linalg.inv(t))

    @staticmethod
    def _is_close_zero(val) -> bool:
        return abs(val) < 1e-5

    @staticmethod
    def _convert_to_real_j_vals(vals):
        ans = deepcopy(vals)
        ans[0] -= pi
        ans[1] = -ans[1]
        ans[2] = -ans[2]
        ans[3] = -ans[3]
        ans[4] -= pi
        return ans

    def solve(self, tcp2base: Transform, seed=None, tcp2end=Transform(), seed_threshold_deg=12) -> Tuple[List, float]:
        T = (tcp2base * tcp2end.inv()).to_mat4x4().reshape(-1)
        sols = []
        pi_2 = pi / 2
        A = T[3] - self.d6 * T[2]
        B = T[7] - self.d6 * T[6]
        R = A * A + B * B

        arccos = acos(self.d4 / sqrt(R))
        arctan = atan2(B, A)
        pos = arctan + arccos
        neg = arctan - arccos
        if self._is_close_zero(pos):
            pos = 0.0

        if self._is_close_zero(neg):
            neg = 0.0

        q1s = [pos + pi_2, neg + pi_2] if not self._is_close_zero(pos - neg) else [pi_2]
        for q1 in q1s:
            numer = (T[3] * sin(q1) - T[7] * cos(q1) - self.d4)
            arccos = acos(numer / self.d6)
            q5s = [0.0] if self._is_close_zero(arccos) else (arccos, -arccos)
            for q5 in q5s:
                c1 = cos(q1)
                s1 = sin(q1)
                c5 = cos(q5)
                s5 = sin(q5)
                q6 = 0
                if not self._is_close_zero(s5):
                    q6 = atan2(s5 / abs(s5) * -(T[1] * s1 - T[5] * c1), s5 / abs(s5) * (T[0] * s1 - T[4] * c1))
                    if self._is_close_zero(q6):
                        q6 = 0.0
                else:
                    logging.warning("[ik_solver] q6 is set to 0")
                c6 = cos(q6)
                s6 = sin(q6)
                x04x = -s5 * (T[2] * c1 + T[6] * s1) - c5 * (s6 * (T[1] * c1 + T[5] * s1) - c6 * (T[0] * c1 + T[4] * s1))
                x04y = c5 * (T[8] * c6 - T[9] * s6) - T[10] * s5
                p13x = self.d5 * (s6 * (T[0] * c1 + T[4] * s1) + c6 * (T[1] * c1 + T[5] * s1)) - self.d6 * (
                        T[2] * c1 + T[6] * s1) + T[3] * c1 + T[7] * s1
                p13y = T[11] - self.d1 - self.d6 * T[10] + self.d5 * (T[9] * c6 + T[8] * s6)
                c3 = (p13x ** 2 + p13y ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2.0 * self.a2 * self.a3)
                if c3 > 1:
                    continue
                arccos = acos(c3)
                q3 = (arccos, -arccos)

                denom = self.a2 ** 2 + self.a3 ** 2 + 2 * self.a2 * self.a3 * c3
                s3 = sin(arccos)
                A = self.a2 + self.a3 * c3
                B = self.a3 * s3

                q2 = [atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom),
                      atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)]
                c23_0 = cos(q2[0] + q3[0])
                s23_0 = sin(q2[0] + q3[0])
                c23_1 = cos(q2[1] + q3[1])
                s23_1 = sin(q2[1] + q3[1])
                q4 = [atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0),
                      atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1)]
                for i in range(1 if self._is_close_zero(q3[0]) else 2):
                    sols.append([q1, q2[i], q3[i], q4[i], q5, q6])

        final = []
        for sol in sols:
            if np.any(np.isnan(sol)):
                continue
            tmp = []
            sol2 = self._convert_to_real_j_vals(sol)
            for val, (hi, lo) in zip(sol2, self.bounds):
                val = wrap2pi(val)
                tmp.append([v for v in (val, val - 2 * pi, val + 2 * pi) if lo <= v <= hi])
                if not tmp[-1]:
                    break
            if tmp[-1]:
                cur = [[]]
                for vs in tmp:
                    cur = [c + [v] for v in vs for c in cur]
                final.extend(cur)

        f2 = []
        if len(final) > 1:
            final.sort()
            p1 = 0
            p2 = 1
            while p2 < len(final):
                while p2 < len(final):
                    ok = False
                    for j in range(6):
                        if not self._is_close_zero(final[p1][j] - final[p2][j]):
                            ok = True
                            break
                    if ok:
                        break
                    else:
                        p2 += 1
                f2.append(final[p1])
                p1 = p2
                p2 += 1
        else:
            f2 = final
        if seed:
            minmax = float('inf')
            for sol in f2:
                found = True
                max_jump = 0
                for _i in range(6):
                    _d = abs(sol[_i] - seed[_i])
                    # print(max_jump, _d)
                    max_jump = max(max_jump, _d)
                    if _d > np.deg2rad(seed_threshold_deg):
                        found = False
                        # print("found invalid: ", rad2deg(abs(sol[_i] - seed[_i])))
                        break
                if found:
                    return [sol], max_jump
                minmax = min(minmax, max_jump)
            return [], minmax
        return f2
