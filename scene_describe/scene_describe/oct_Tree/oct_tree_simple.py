from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple
import math

from oct_Tree.scene_data import tools
import oct_Tree.test as t


class ArtiPotentialPoint:
    def __init__(self, attribute, depth, index_array):
        self.depth = depth
        self.attribute = attribute  # 势峰中心还是势阱中心
        self.index_array = index_array
        # attribute表示虚拟力朝向该势点还是背离该势点，0是背离，1是朝向
        self.center = compute_center_from_index(self.index_array)
        self.fi = 0.08  # 计算虚拟力的距离阈值
        self.n = 2  # 计算虚拟力的衰减指数

    def get_force(self, points):
        fi = self.fi
        point = np.array(points)
        direction = np.array(self.center) - point  # 现在方向指向center
        dis = np.linalg.norm(direction)  # 三维欧式距离
        if dis == 0:
            return np.array([0, 0, 0], dtype=float)  # 势点所受的虚拟力
        dir_n = direction / dis  # 归一后的单位方向向量
        depth = self.depth
        w = math.exp(1 + 1 / depth) / (math.exp(1 + 0.5) + math.exp(1 + 0.33) + math.exp(1 + 0.25))  # 层数决定的权重
        if dis > fi:
            return np.array([0, 0, 0], dtype=float)
        F_abs_1 = w * math.pow(dis, 1 / self.n)  # 引力
        F_abs_0 = w / math.pow(dis, self.n)  # 斥力
        if self.attribute:
            return F_abs_1 * dir_n
        else:
            return F_abs_0 * (-1) * dir_n


class OctreeNode:
    def __init__(self, index, depth, parent = None):
        self.size = float(0.16 / depth)  # 节点大小 (立方体的边长)第一层是root
        self.children = [None] * 8  # 八个孩子节点
        self.points = []  # 存储点的列表
        self.depth = depth  # 当前节点的深度
        self.parent = parent
        self.index = index
    def __str__(self):
        return f"Node: {self.index} at depth {self.depth}"
    def compute_p(self):
        num_of_voelx = len(self.points)
        p = (num_of_voelx * 0.02 * 0.02) / (self.size * self.size)
        return p

    def compute_rou(self):
        i = 0
        rou_son = 0
        if self.depth == 3:
            for item in self.children:
                if item:
                    i += 1
            return i / 8
        for item in self.children:
            if item:
                i += 1
                rou_son += item.compute_rou()
        w1 = 0.6
        w2 = 0.4
        rou = w1 * i / 8 + w2 * rou_son / i
        return rou
    def show_tree(self):
        print(self)
        for child in self.children:
            if child:
                child.show_tree()

def comput_index(parent, son):
    point = son
    parent = parent
    relative_pos = np.array(point) - np.array(parent)
    x = relative_pos[0]
    y = relative_pos[1]
    z = relative_pos[2]
    if x > 0:
        if y > 0:
            if z > 0:
                return 0
            else:
                return 4
        else:
            if z > 0:
                return 3
            else:
                return 7
    else:
        if y > 0:
            if z > 0:
                return 1
            else:
                return 5
        else:
            if z > 0:
                return 2
            else:
                return 6


def insert_node(root: OctreeNode, point):
    index = compute_index_array(point)
    root.points.append(point)
    if root.children[index[0]]:
        if root.children[index[0]].children[index[1]]:
            if root.children[index[0]].children[index[1]].children[index[2]]:
                root.children[index[0]].children[index[1]].children[index[2]].points.append(point)
            else:
                root.children[index[0]].children[index[1]].children[index[2]] = OctreeNode(index=index[2], depth=4,
                                                                                           parent=root.children[
                                                                                               index[0]].children[
                                                                                               index[1]])
                root.children[index[0]].children[index[1]].children[index[2]].points.append(point)
        else:
            root.children[index[0]].children[index[1]] = OctreeNode(index=index[1], depth=3,
                                                                    parent=root.children[index[0]])
    else:
        root.children[index[0]] = OctreeNode(index=index[0], depth=2, parent=root)

        root.children[index[0]].children[index[1]] = OctreeNode(index=index[1], depth=3, parent=root.children[index[0]])
        root.children[index[0]].children[index[1]].children[index[2]] = OctreeNode(index=index[2], depth=4, parent=
        root.children[index[0]].children[index[1]])
    root.children[index[0]].points.append(point)
    root.children[index[0]].children[index[1]].points.append(point)


def compute_index_array(point):
    ##此处的point坐标是把中心点定位原点后的相对坐标
    layer = {0: [1, 1, 1], 1: [-1, 1, 1], 2: [-1, -1, 1], 3: [1, -1, 1],
             4: [1, 1, -1], 5: [-1, 1, -1], 6: [-1, -1, -1], 7: [1, -1, -1]}  ##第二层乘以4，第三层乘以2
    index = []
    index.append(comput_index([0, 0, 0], point))
    center = 0.04 * np.array(layer[index[0]])
    index.append(comput_index(center, point))
    center = 0.02 * np.array(layer[index[1]]) + center
    index.append(comput_index(center, point))
    index = np.array(index)
    return index


def compute_layer_sim(node_empir: OctreeNode, node_current: OctreeNode):
    sim = 0
    nodes_empir = node_empir.children
    nodes_current = node_current.children
    depth = node_current.depth
    if depth == 4:
        return 1
    num = 8
    for i in range(8):
        if nodes_empir[i] and nodes_current[i]:
            sim += compute_layer_sim(nodes_empir[i], nodes_current[i])
        if not nodes_current[i] and not nodes_empir[i]:
            sim += 1
        if (not nodes_current[i] and nodes_empir[i]) or (nodes_current[i] and not nodes_empir[i]):
            if depth == 3:
                tep_sim = 0
            else:
                if nodes_current[i]:
                    p_current = nodes_current[i].compute_p()
                    rou = nodes_current[i].compute_rou()
                else:
                    p_current = 0
                if nodes_empir[i]:
                    p_empir = nodes_empir[i].compute_p()
                    rou = nodes_empir[i].compute_rou()
                else:
                    p_empir = 0
                delta = abs(p_current - p_empir)

                w1 = 0.5
                w2 = 0.5
                tep_sim = w1 * delta + w2 * rou
                if depth == 3: tep_sim = 0
            sim += tep_sim
    return sim / 8


def compute_sim(root1: OctreeNode, root2: OctreeNode):
    node_1 = root1
    node_2 = root2
    return compute_layer_sim(node_1, node_2)


def compute_potential(root_history: OctreeNode, root_current: OctreeNode):  # 返回一个势点列表
    potential_points = []
    node_empir = root_history
    node_current = root_current
    nodes_empir = node_empir.children
    nodes_current = node_current.children
    depth = node_current.depth
    if depth == 4:
        return potential_points

    for i in range(8):
        if not nodes_current[i] and nodes_empir[i]:  # 当前场景该节点没有障碍物，历史经验节点有
            # 应该指向该节点，即1
            temp = nodes_empir[i]
            dep = temp.depth
            index_array = []
            index_array.append(i)
            while temp.parent:
                if temp.parent.depth != 1:
                    index_array.append(temp.parent.index)
                temp = temp.parent
            index_array.reverse()
            potential_points.append(ArtiPotentialPoint(1, dep, index_array))
        if nodes_current[i] and not nodes_empir[i]:  # 当前场景该节点有障碍物，历史经验节点没有
            # 应该背离该节点，即0
            temp = nodes_current[i]
            dep = temp.depth
            index_array = []
            index_array.append(i)
            while temp.parent:
                if temp.parent.depth != 1:
                    index_array.append(temp.parent.index)
                temp = temp.parent
            index_array.reverse()
            potential_points.append(ArtiPotentialPoint(1, dep, index_array))
        if nodes_empir[i] and nodes_current[i]:
            potential_points.extend(compute_potential(nodes_empir[i], nodes_current[i]))

    return potential_points


def compute_center_from_index(index_array):
    # 根据一个节点的索引序列来计算该节点中心位置
    distance = 0.04
    array = index_array
    relative = np.array([0, 0, 0], dtype=float)
    layer = {0: [1, 1, 1], 1: [-1, 1, 1], 2: [-1, -1, 1], 3: [1, -1, 1],
             4: [1, 1, -1], 5: [-1, 1, -1], 6: [-1, -1, -1], 7: [1, -1, -1]}  # 第二层乘以4，第三层乘以2
    for index in array:
        relative += distance * np.array(layer[index])
        distance = distance / 2
    return relative


def compute_total_force(potential_points, point):  # point是np数组,potential_points是由compute_potential()得到的势点列表
    potential_center = potential_points
    force = np.array([0, 0, 0], dtype=float)
    for item in potential_center:
        force += item.get_force(point)
    return force


if __name__ == "__main__":
    print(compute_index_array([0.02, -0.07, 0.09]))
    print(compute_index_array([0.15, -0.04, -0.06]))
    print(compute_index_array([-0.01, -0.1, 0.09]))
    print(compute_index_array([-0.12, 0.14, -0.02]))
    excel_path = './scene_data/data_of_scene1.xlsx'
    points = tools.read_3d_data(excel_path)
    low = [0.46, 0, 0.48]
    up = [0.7, 0.2, 0.68]
    center = np.array([0.58, 0.08, 0.57])
    scene_1 = tools.filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    # 生成八叉树
    root_emp = OctreeNode(0, depth=1, parent=None)
    for item in scene_1:
        insert_node(root_emp, item)
    # 至此得到了示例八叉树root_emp
    print("-----------------------------------------------------------------------------------------------------------")
    root_emp.show_tree()
    print("-----------------------------------------------------------------------------------------------------------")
    file_path = f"scene_data/data_of_scene/data_of_scene{45}.txt"
    root_current = t.generate_octree_from_txt(file_path)

    sim = compute_sim(root_emp, root_current)
    # sim2 = compute_layer_similarity(root_emp, root_current)
    print("sim = :", sim)
    # print("sim2 = :", sim2)
    poential = compute_potential(root_history=root_emp, root_current=root_current)
    print(len(poential))
    for item in poential:
        print(item.index_array,'[', item.center[0], ',', item.center[1], ',', item.center[2], '],')
    print(compute_total_force(poential, np.array([0.53115991, 0.09719978, 0.53595373], dtype=float) - center))
    print(compute_total_force(poential, np.array([0.56462763, 0.12902873, 0.57918368], dtype=float) - center))
    print(compute_total_force(poential, np.array([0.52360095, 0.02124487, 0.52451799], dtype=float) - center))
    for item in root_current.points:
        print(item[0] + center[0], ' ', item[1] + center[1], ' ', item[2] + center[2])
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for item in root_current.points:
        print(item[0] + center[0], ' ', item[1] + center[1], ' ', item[2] + center[2])

    """
    max_value = np.max(np.array(sims))
    max_index = np.argmax(np.array(sims))
    print("最大相似度:", max_value)
    print("最相似场景下标:", max_index)
    points_sim = roots[43].points
    for item in points_sim:
        item = np.array(item) + center
        print(item[0], ' ', item[1], ' ', item[2])

    # 打印某个场景的点坐标，帮助在rviz复现
    points_sim = roots[231].points
    for item in points_sim:
        item = np.array(item) + center
        print(item[0], ' ', item[1], ' ', item[2])
    """

