import math

import numpy
import numpy as np
from typing import Optional,List,Tuple
from oct_Tree.scene_data import tools
import tools as t
# 设置全局变量，代表局部区域正方体边长和体素边长

env_size = 0.16
voxel_size = 0.02
max_depth = 4


class OctreeNode:
    def __init__(
            self,
            index: int = -1,
            depth: int = 1,
            parent: Optional['OctreeNode'] = None,
            base_size: float = env_size,
            voxel_size: float = voxel_size,
            density_weights: Tuple[float, float] = (0.6, 0.4)
    ):
        """
        八叉树节点类

        Args:
            index: 当前节点在父节点中的索引(0-7)
            depth: 节点深度(根节点为0)
            parent: 父节点引用
            base_size: 根节点的大小(默认0.16)
            voxel_size: 体素大小(默认0.02)
            density_weights: rou计算的权重(w1, w2)
        """
        self.index = index
        self.depth = depth
        self.parent = parent
        self.children = [None] * 8
        self.points = []
        self._base_size = base_size
        self._voxel_size = voxel_size
        self._density_weights = density_weights

        # 计算节点大小(遵循2的幂次递减)
        self.size = self._compute_node_size()

    def _compute_node_size(self) -> float:
        """计算当前节点的大小"""
        if self.depth == 1:  # 根节点特殊处理
            return self._base_size
        return self._base_size / (2 ** (self.depth-1))

    def compute_p(self) -> float:
        """
        计算当前节点的p值(占用率)

        Returns:
            当前节点中体素占据的空间比例(0-1)
        """

        points = self.points
        points = [p for p in points if np.all(np.abs(p) <= 0.5*env_size)]
        s = voxel_size
        voxel_set = set()

        for x, y, z in points:
            # 使用 math.floor，确保对负坐标也正确
            i = math.floor(x / s)
            j = math.floor(y / s)
            k = math.floor(z / s)
            voxel_set.add((i, j, k))
        #
        # print("所有栅格数量:",len(voxel_set))

        voxel_area = self._voxel_size ** 2
        node_area = self.size ** 2
        # print("当前节点的面积:",node_area)
        # print("当前节点的体素面积:",voxel_area)
        return (len(voxel_set) * voxel_area) / node_area

    def compute_rou(self, depth_threshold: int = max_depth-1) -> float:
        """
        计算当前节点的rou值(密度)

        Args:
            depth_threshold: 计算密度的深度阈值(默认为3)该值需要根据局部区域size和voxel_size来调整，本质是最大深度d-1

        Returns:
            当前节点的密度值(0-1)
        """

        # 统计有效子节点数量
        valid_children = [child for child in self.children if child is not None]
        num_children = len(valid_children)

        # 对于叶子节点或者超过深度阈值的节点
        if self.depth == depth_threshold or not valid_children:
            return num_children / 8

        # 递归计算子节点密度
        children_rou_sum = sum(child.compute_rou(depth_threshold) for child in valid_children)

        # 应用权重计算综合密度
        w1, w2 = self._density_weights
        return w1 * (num_children / 8) + w2 * (children_rou_sum / num_children)

    def is_leaf(self) -> bool:
        """判断是否是叶子节点"""
        return all(child is None for child in self.children)

    def __str__(self):
        return f"Node: {self.index} at depth {self.depth}"

    def show_tree(self):
        print(self)
        for child in self.children:
            if child:
                child.show_tree()


def compute_index(parent, son):
    """
    以 parent 为原点，计算 son 在八个象限中的位置（0-7）。
    编码规则：
      - 第一层（z>0）：0(+++), 1(-++), 2(--+), 3(+-+)
      - 第二层（z<0）：4(++-), 5(-+-), 6(---), 7(+--)

    Args:
        parent: 参考原点坐标（可迭代或np.array）。
        son: 待判断点坐标（可迭代或np.array）。

    Returns:
        int: 象限编号（0-7）。
    """
    dic = {"[1, 1, 1]": 0, "[0, 1, 1]": 1, "[0, 0, 1]": 2, "[1, 0, 1]": 3,"[1, 1, 0]": 4, "[0, 1, 0]": 5, "[0, 0, 0]": 6, "[1, 0, 0]": 7}
    parent = np.asarray(parent)
    son = np.asarray(son)
    relative_pos = son - parent

    x_sign, y_sign, z_sign = (relative_pos > 0).astype(int)
    # print(x_sign, y_sign, z_sign)
    index = dic[str([x_sign, y_sign, z_sign])]
    return int(index)


def compute_index_array(point, total_size=env_size, min_size=voxel_size, origin=None):
    """
    计算点在分层立方体系统中的索引数组。

    Args:
        point: 要定位的点坐标 (3D)
        total_size: 立方体总边长 (默认16)
        min_size: 最小子立方体边长 (默认2)
        origin: 立方体系统中心点 (默认[0,0,0])

    Returns:
        np.array: 各层的索引组成的数组
    """
    if origin is None:
        origin = np.array([0., 0., 0.])
    else:
        origin = np.array(origin, dtype=float)

    point = np.array(point, dtype=float)
    relative_point = point - origin

    # 确定细分层数
    n_layers = int(np.log2(total_size / min_size))
    if n_layers <= 0:
        return np.array([compute_index(origin, point)])

    # 定义八分方向向量

    octants = {0: [1, 1, 1], 1: [-1, 1, 1], 2: [-1, -1, 1], 3: [1, -1, 1],
             4: [1, 1, -1], 5: [-1, 1, -1], 6: [-1, -1, -1], 7: [1, -1, -1]}
    index_array = []
    current_center = origin.copy()

    for layer in range(n_layers):
        # 计算当前层级的索引
        idx = compute_index(current_center, relative_point + origin)
        index_array.append(idx)

        # 如果是最后一层，不再计算下一层中心
        if layer == n_layers - 1:
            break

        # 计算下一层中心偏移量

        current_center += np.array(octants[idx]) * total_size/(2 ** (layer + 2))

    return np.array(index_array)


def insert_node(root: OctreeNode, point, maxdepth: int = max_depth-1):
    """
    向八叉树中插入一个点

    Args:
        root: 八叉树根节点
        point: 要插入的点坐标
        max_depth: 最大插入深度(默认为3)
        :param point:
        :param root:
        :param maxdepth:
    """
    if not root:
        raise ValueError("Root node cannot be None")

    # 计算插入路径
    index_array = compute_index_array(point)
    if len(index_array) < maxdepth:
        max_depth = len(index_array)

    # 沿着路径插入到适当层级
    current = root
    current.points.append(point)  # 每个途经节点都存储该点

    for depth in range(maxdepth):
        idx = index_array[depth]

        if not current.children[idx]:
            # 创建新节点
            current.children[idx] = OctreeNode(
                index=idx,
                depth=depth + 2,
                parent=current
            )

        current = current.children[idx]
        current.points.append(point)


def compute_layer_similarity(
        empirical_node: OctreeNode,
        current_node: OctreeNode,
        weights: Tuple[float, float] = (0.5, 0.5),
        max_depth: int = max_depth,
        similarity_threshold: float = 0,
) -> float:
    """
    计算两个八叉树节点在指定层级的相似度

    Args:
        empirical_node: 经验数据对应的八叉树节点
        current_node: 当前数据对应的八叉树节点
        weights: 相似度计算的权重因子(w1, w2)
        max_depth: 最大计算深度(默认4)
        similarity_threshold: 叶子节点的相似度阈值(默认0)

    Returns:
        当前层级的相似度得分(0-1之间)
    """

    def get_node_metrics(node: Optional[OctreeNode]) -> Tuple[float, float]:
        """获取节点的p值和rou值"""
        if node is None:
            return 0.0, 0.0
        return node.compute_p(), node.compute_rou()

    if current_node.depth == max_depth:
        return similarity_threshold

    similarity_sum = 0.0
    w1, w2 = weights

    for i in range(8):
        emp_child = empirical_node.children[i]
        cur_child = current_node.children[i]

        # 情况1: 两个子节点都存在
        if emp_child and cur_child:
            similarity_sum += compute_layer_similarity(
                emp_child, cur_child, weights, max_depth, similarity_threshold
            )

        # 情况2: 两个子节点都不存在
        elif not emp_child and not cur_child:
            similarity_sum += 1.0

        # 情况3: 只有一个子节点存在
        else:
            if current_node.depth == max_depth - 1:
                # 最底下一层特殊处理
                temp_sim = similarity_threshold
            else:
                # 计算差异度量
                p_cur, rou_cur = get_node_metrics(cur_child)
                p_emp, rou_emp = get_node_metrics(emp_child)

                delta_p = abs(p_cur - p_emp)
                rou = rou_cur if cur_child else rou_emp

                temp_sim = w1 * delta_p + w2 * rou

            similarity_sum += temp_sim

    return similarity_sum / 8


def compute_similarity(root1: OctreeNode, root2: OctreeNode):
    return compute_layer_similarity(root1, root2)


def generate_octree_from_txt(path):
    # 从一个txt文档里面读取点，然后过滤到指定栅格，然后计算相对坐标，最好生成八叉树，返回根节点
    points = t.read_data_from_txt(path)
    low = [0.46, 0, 0.48]
    up = [0.7, 0.2, 0.68]
    center = np.array([0.58, 0.08, 0.57])
    scene = t.filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    # print("当前场景的点数：", len(scene))
    root = OctreeNode(0, depth=1, parent=None)
    for item in scene:
        insert_node(root, item)
    return root

def get_best_experience(base_size, current_scene):
    """
    计算当前场景的经验数据
    :param base_size: 基准大小
    :param current_scene: 当前场景
    :return: 经验数据
    """
    roots = []  # 存储根节点
    sims = []  # 存储所有相似度
    delta_p = []
    for i in range(base_size):
        file_path = f"scene_data/data_of_scene/data_of_scene{i}.txt"
        roots.append(generate_octree_from_txt(file_path))
    for item in roots:
        sims.append(compute_similarity(current_scene, item))
        delta_p.append(abs(current_scene.compute_p() - item.compute_p()))
    sim_max = 0
    max_index= 0
    for j in range(base_size):
        if sims[j] > sim_max:
            sim_max = sims[j]
            max_index = j
    print("max:", max_index, sim_max)
if  __name__ == '__main__':
    excel_path = './scene_data/baseline_scene.xlsx'
    points = t.read_3d_data(excel_path)
    low = [0.46, 0, 0.48]
    up = [0.7, 0.2, 0.68]
    center = np.array([0.58, 0.08, 0.57])
    scene_1 = t.filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    # 生成八叉树
    root_emp = OctreeNode(0, depth=1, parent=None)
    for item in scene_1:
        insert_node(root_emp, item)
    # 下面得到其他txt中的测试八叉树数据
    roots = []
    complex = []
    for i in range(1000):
        file_path = f"./scene_data/data_of_scene/data_of_scene{i}.txt"
        roots.append(generate_octree_from_txt(file_path))
        complex.append(roots[i].compute_p())
        # print(i)
        # print("当前场景的p值：", roots[i].compute_p())
    complex = np.array(complex)
    # 找出complex中接近0.1，0.2，0.3，0.4，0.5，0.6，0.7，0.8的下标
    for i in range(len(complex)):
        # if abs(complex[i] - 0.1) < 0.01:
        #     print("10%:", i)
        # if abs(complex[i] - 0.2) < 0.01:
        #     print("20%:", i)
        # if abs(complex[i] - 0.3) < 0.01:
        #     print("30%:", i)
        # if abs(complex[i] - 0.4) < 0.01:
        #     print("40%:", i)
        # if abs(complex[i] - 0.5) < 0.01:
        #     print("50%:", i)
        # if abs(complex[i] - 0.6) < 0.01:
        #     print("60%:", i)
        if abs(complex[i] - 0.7) < 0.02:
            print("70%:", i)
        if abs(complex[i] - 0.8) < 0.02:
            print("80%:", i)
        # if abs(complex[i] - 0.9) < 0.05:
        #     print("90%:", i)
    # print("当前场景的p值：",complex)
    # print("最大值：",complex.max())
    # print("最大值的下标：",complex.argmax())
    # print("最小值：",complex.min())
    # print("平均值：",complex.mean())
    # print(root_emp.points)

    # roots = []  # 存储根节点
    # num_of_scene = 1000  # 场景数据数量
    # sims = []  # 存储所有相似度
    # delta_p = []
    # num_of_strong = 0
    # num_of_mid = 0
    # for i in range(1000):
    #     file_path = f"scene_data/data_of_scene/data_of_scene{i}.txt"
    #     roots.append(generate_octree_from_txt(file_path))
    # for item in roots:
    #     sims.append(compute_similarity(root_emp, item))
    #     delta_p.append(abs(root_emp.compute_p() - item.compute_p()))
    # sim_max = 0
    # max_index = 0
    # for i in range(1000):
    #     if sims[i]>sim_max:
    #         sim_max = sims[i]
    #         max_index = i
    #     if abs(sims[i] -0.1)<0.01 and delta_p[i]<0.1:
    #         print("10%:",i)
    #     if abs(sims[i] -0.2)<0.01 and delta_p[i]<0.1:
    #         print("20%:",i)
    #     if abs(sims[i] -0.3)<0.01 and delta_p[i]<0.1:
    #         print("30%:",i)
    #     if abs(sims[i] -0.4)<0.01 and delta_p[i]<0.1:
    #         print("40%:",i)
    #     if abs(sims[i] -0.5)<0.01 and delta_p[i]<0.1:
    #         print("50%:",i)
    #     if abs(sims[i] -0.6)<0.01 and delta_p[i]<0.1:
    #         print("60%:",i)
    #     if abs(sims[i] -0.7)<0.01 and delta_p[i]<0.1:
    #         print("70%:",i)
    #     if abs(sims[i] -0.8)<0.01 and delta_p[i]<0.1:
    #         print("80%:",i)
    #     if abs(sims[i] -0.9)<0.01 and delta_p[i]<0.1:
    #         print("90%:",i)
    # print("max:",max_index,sim_max)
    # print("-----------------------------------------------------------------------------------------------------------")
    # # sims = np.array(sims)
    # # print(sims)