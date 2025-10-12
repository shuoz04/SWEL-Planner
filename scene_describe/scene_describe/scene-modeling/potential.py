import pickle
# from GMM.GMM import WeightedGMM
from oct_tree import OctreeNode, insert_node, compute_similarity, generate_octree_from_txt
import math
from typing import List, Union, Tuple, Dict
import numpy as np
from oct_Tree.scene_data import tools

env_size = 0.16
voxel_size = 0.02
max_depth = 4
# class ArtiPotentialPoint:
#     def __init__(self, attribute, depth, index_array):
#         self.depth = depth
#         self.attribute = attribute  # 势峰中心还是势阱中心
#         self.index_array = index_array
#         # attribute表示虚拟力朝向该势点还是背离该势点，0是背离，1是朝向
#         self.center = compute_center(self.index_array)
#         self.fi = 0.08  # 计算虚拟力的距离阈值
#         self.n = 2  # 计算虚拟力的衰减指数
#
#     def get_force(self, points):
#         """
#         计算势点对于某个点point的虚拟力
#         :param points:
#         :return:
#         """
#         fi = self.fi
#         point = np.array(points)
#         direction = np.array(self.center) - point  # 现在方向指向center
#         dis = np.linalg.norm(direction)  # 三维欧式距离
#         if dis == 0:
#             return np.array([0, 0, 0], dtype=float)  # 势点所受的虚拟力
#         dir_n = direction / dis  # 归一后的单位方向向量
#         depth = self.depth
#         w = math.exp(1 + 1 / depth) / (math.exp(1 + 0.5) + math.exp(1 + 0.33) + math.exp(1 + 0.25))  # 层数决定的权重
#         if dis > fi:
#             return np.array([0, 0, 0], dtype=float)
#         F_abs_1 = w * math.pow(dis, 1 / self.n)  # 引力
#         F_abs_0 = w / math.pow(dis, self.n)  # 斥力
#         if self.attribute:
#             return F_abs_1 * dir_n
#         else:
#             return F_abs_0 * (-1) * dir_n
from sklearn.cluster import KMeans

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
        return loglik





class ArtiPotentialPoint:
    """
    人工势场点模型，用于生成引力/斥力场

    根据深度和空间位置计算对目标点的虚拟力：
    - 势峰中心(attribute=0): 产生斥力（向外推）
    - 势阱中心(attribute=1): 产生引力（向内拉）
    """

    # 类常量（全大写命名）
    DEFAULT_DISTANCE_THRESHOLD = 0.08  # 默认力作用范围 (fi)
    DEFAULT_DECAY_EXPONENT = 2  # 默认力衰减指数 (n)

    # 预计算的深度权重分母（避免每次计算）
    _DEPTH_WEIGHT_DENOMINATOR = (
            math.exp(2.5) + math.exp(1.33) + math.exp(1.25)  # exp(1+0.5) + exp(1+0.33) + exp(1+0.25)
    )

    def __init__(
            self,
            attribute: int,
            depth: int,
            index_array: List[int],
            force_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
            decay_exponent: float = DEFAULT_DECAY_EXPONENT
    ):
        """
        初始化势场点

        Args:
            attribute: 势场类型 (0: 斥力源, 1: 引力源)
            depth: 八叉树深度 (影响权重)
            index_array: 八叉树索引序列
            force_threshold: 力作用的最大距离
            decay_exponent: 力衰减指数
        """
        self.depth = depth
        self.attribute = self._validate_attribute(attribute)
        self.index_array = index_array
        self.center = self._compute_center()
        self.fi = force_threshold
        self.n = decay_exponent

    @staticmethod
    def _validate_attribute(attr: int) -> int:
        """验证并返回合法的attribute值"""
        if attr not in (0, 1):
            raise ValueError("Attribute must be 0 (repulsive) or 1 (attractive)")
        return attr

    def _compute_center(self) -> np.ndarray:
        """计算势点的中心坐标 (使用八叉树索引)"""
        return compute_center(self.index_array)  # 使用前面优化过的函数

    def _calculate_depth_weight(self) -> float:
        """计算基于深度的动态权重"""
        return math.exp(1 + 1 / self.depth) / self._DEPTH_WEIGHT_DENOMINATOR

    def get_force(
            self,
            points: Union[np.ndarray, List[float], Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        计算对目标点的虚拟力向量

        Args:
            points: 目标点坐标 (可以是任意三维可迭代对象)

        Returns:
            np.ndarray: 三维力向量 (零向量表示超出作用范围)

        Raises:
            ValueError: 如果输入不是三维坐标
        """
        target_point = np.asarray(points, dtype=float)
        if target_point.shape != (3,):
            raise ValueError("Input must be a 3D coordinate")

        direction = self.center - target_point
        distance = np.linalg.norm(direction)

        # 零距离或超出作用范围的情况
        if math.isclose(distance, 0) or distance > self.fi:
            return np.zeros(3)

        normalized_dir = direction / distance
        weight = self._calculate_depth_weight()

        if self.attribute == 1:  # 引力
            force_magnitude = weight * (distance ** (1 / self.n))
            return force_magnitude * normalized_dir
        else:  # 斥力
            force_magnitude = weight / (distance ** self.n)
            return -force_magnitude * normalized_dir

    def __repr__(self) -> str:
        """对象表示(用于调试)"""
        return (
            f"ArtiPotentialPoint(attribute={self.attribute}, "
            f"depth={self.depth}, center={self.center.tolist()})"
        )


def compute_center(
        index_array: List[int],
        base_distance: float = env_size / 4
) -> np.ndarray:
    """
    根据节点的索引序列计算该节点在八叉树中的中心相对位置

    按照八叉树节点索引的层级关系，递归计算中心坐标。
    每个索引对应一个八分象限的方向向量，逐层缩小距离。

    Args:
        index_array: 节点的索引序列 (列表形式，每个元素是0-7的整数)
        base_distance: 初始距离（根节点下一层节点的半边长\半径）

    Returns:
        np.ndarray: 中心点的相对坐标 (3D向量)

    Raises:
        ValueError: 如果索引值不在0-7范围内
    """
    # 定义每个索引对应的方向向量（单位向量）
    INDEX_TO_DIRECTION: Dict[int, Tuple[float, float, float]] = {
        0: (1, 1, 1),  # 第一象限（右上前）
        1: (-1, 1, 1),  # 第二象限（左上前）
        2: (-1, -1, 1),  # 第三象限（左下前）
        3: (1, -1, 1),  # 第四象限（右下前）
        4: (1, 1, -1),  # 第五象限（右上后）
        5: (-1, 1, -1),  # 第六象限（左上后）
        6: (-1, -1, -1),  # 第七象限（左下后）
        7: (1, -1, -1)  # 第八象限（右下后）
    }

    # 输入验证
    for index in index_array:
        if index not in INDEX_TO_DIRECTION:
            raise ValueError(f"Invalid octree index {index}. Must be 0-7.")

    relative_position = np.zeros(3, dtype=float)
    current_distance = base_distance

    for index in index_array:
        direction = np.array(INDEX_TO_DIRECTION[index])
        relative_position += current_distance * direction
        current_distance /= 2  # 每层距离减半

    return relative_position


def compute_potential(
        root_history: OctreeNode,
        root_current: OctreeNode,
        max_depth: int = max_depth
) -> List[ArtiPotentialPoint]:
    """
    通过比较历史八叉树和当前八叉树，生成势点列表

    算法逻辑：
    1. 递归比较两个八叉树的对应节点
    2. 当历史节点存在障碍而当前节点不存在时，生成引力点(吸引向历史安全区域)
    3. 当当前节点存在障碍而历史节点不存在时，生成斥力点(排斥新障碍物)
    4. 仅处理到指定最大深度(max_depth)的节点

    Args:
        root_history: 历史经验八叉树根节点
        root_current: 当前场景八叉树根节点
        max_depth: 最大处理深度 (默认4层)

    Returns:
        List[ArtiPotentialPoint]: 生成的势场点列表

    Raises:
        ValueError: 如果输入的根节点深度不一致
    """
    # 输入验证
    if root_history.depth != root_current.depth:
        raise ValueError("History and current roots must have same depth")

    potential_points: List[ArtiPotentialPoint] = []

    # 终止条件
    if root_current.depth >= max_depth:
        return potential_points

    # 获取子节点列表（处理可能为None的情况）
    nodes_history = root_history.children if root_history.children else []
    nodes_current = root_current.children if root_current.children else []

    # 比较8个子节点
    for i in range(8):
        # 边界检查
        if i >= len(nodes_history) or i >= len(nodes_current):
            continue

        node_history = nodes_history[i]
        node_current = nodes_current[i]

        # Case 1: 历史有障碍，当前无障碍 → 引力点
        if node_history and not node_current:
            potential_points.append(_create_attraction_point(node_history, i))

        # Case 2: 当前有障碍，历史无障碍 → 斥力点
        elif node_current and not node_history:
            potential_points.append(_create_repulsion_point(node_current, i))

        # Case 3: 两者都有障碍 → 递归处理
        elif node_history and node_current:
            potential_points.extend(
                compute_potential(node_history, node_current, max_depth)
            )

    return potential_points


def _create_attraction_point(node: OctreeNode, child_index: int) -> ArtiPotentialPoint:
    """为安全区域创建引力势点"""
    index_array = _build_index_array(node, child_index)
    return ArtiPotentialPoint(
        attribute=1,  # 引力
        depth=node.depth,
        index_array=index_array
    )


def _create_repulsion_point(node: OctreeNode, child_index: int) -> ArtiPotentialPoint:
    """为新障碍物创建斥力势点"""
    index_array = _build_index_array(node, child_index)
    return ArtiPotentialPoint(
        attribute=0,  # 斥力
        depth=node.depth,
        index_array=index_array
    )


def _build_index_array(node: OctreeNode, child_index: int) -> List[int]:
    """
    构建从根到当前节点的索引路径

    Args:
        node: 当前节点
        child_index: 在父节点中的索引

    Returns:
        排序后的索引数组 (从根到当前节点，不包含根节点的子节点索引)
    """
    indices = [child_index]
    current = node

    while current.parent and current.parent.depth != 1:
        indices.append(current.parent.index)
        current = current.parent

    return list(reversed(indices))


def compute_total_force(
        potential_points: List[ArtiPotentialPoint],
        point: Union[np.ndarray, List[float], Tuple[float, float, float]],
        force_threshold: float = 1e-6
) -> np.ndarray:
    """
    计算给定点在所有势场中的合虚拟力

    Args:
        potential_points: ArtiPotentialPoint对象列表 (来自compute_potential)
        point: 目标点坐标 (x,y,z)
        force_threshold: 力大小的最小阈值 (默认1e-6，避免微小力累积)

    Returns:
        np.ndarray[float]: [Fx, Fy, Fz] 三维合力向量

    Raises:
        ValueError: 如果输入点不是有效三维坐标
        TypeError: 如果potential_points包含错误类型
    """
    # 输入验证和标准化
    target_point = _validate_and_normalize_point(point)
    _validate_potential_points(potential_points)

    # 预分配结果数组 (比累积更高效)
    total_force = np.zeros(3, dtype=np.float64)

    for potential in potential_points:
        try:
            force = potential.get_force(target_point)
            # 只添加超过阈值的力 (噪声过滤)
            if np.linalg.norm(force) > force_threshold:
                total_force += force
            else:
                total_force += np.array([1e-6, 1e-6, 1e-6], dtype=float)
        except AttributeError:
            raise TypeError(f"Invalid potential point type: {type(potential)}")

    return total_force


def _validate_and_normalize_point(
        point: Union[np.ndarray, List[float], Tuple[float, float, float]]
) -> np.ndarray:
    """验证并标准化输入点为三维numpy数组"""
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"Point must be 3D coordinate, got shape {p.shape}")
    return p


def _validate_potential_points(points: List[ArtiPotentialPoint]) -> None:
    """检查势点列表有效性"""
    if not isinstance(points, list):
        raise TypeError(f"potential_points must be list, got {type(points)}")
    if not all(hasattr(p, 'get_force') for p in points):
        raise TypeError("All potential points must have get_force method")

def get_gmm_finetuning(pkl_path,scene_num,step_size,current_scene,new_pkl_path):
    center = np.array([0.397, 0.0833, 0.583])

    with open(pkl_path, 'rb') as f:
        gmm_pos = pickle.load(f)
    print(gmm_pos.mu)
    print(gmm_pos.pi)
    file_path = f"scene_data/data_of_scene/data_of_scene{scene_num}.txt"
    root_his = generate_octree_from_txt(file_path)
    poential = compute_potential(root_history=root_his, root_current=current_scene)
    force_1 = compute_total_force(poential, np.array(gmm_pos.mu[0], dtype=float) - center)
    force_2 = compute_total_force(poential, np.array(gmm_pos.mu[1], dtype=float) - center)
    force_3 = compute_total_force(poential, np.array(gmm_pos.mu[2], dtype=float) - center)

    force_1 = np.array(force_1)
    force_2 = np.array(force_2)
    force_3 = np.array(force_3)

    force_1 = force_1 / np.linalg.norm(force_1)
    force_2 = force_2 / np.linalg.norm(force_2)
    force_3 = force_3 / np.linalg.norm(force_3)
    print(force_1)
    print(force_2)
    print(force_3)
    gmm_pos.mu[0] += step_size * force_1
    gmm_pos.mu[1] += step_size * force_2
    gmm_pos.mu[2] += step_size * force_3
    # 保存修改后的gmm
    with open(new_pkl_path, 'wb') as f:
        pickle.dump(gmm_pos, f)


if __name__ == '__main__':
    # 得到示例数据的八叉树根节点
    excel_path = 'scene_data/baseline_scene.xlsx'
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
    # 读取当前场景数据
    dict = {20:31,30:239,40:32,50:964,60:40,70:900,80:64,90:639}
    for j in range(3,4):
        for i in range(1,2):
            pkl_path = "../GMM/data/GMM_model/pos_model/gmm_{}%_pos.pkl".format(i*10)
            scene_num = dict[i*10]
            step_size = j * 0.01
            new_pkl_path = "../GMM/data/GMM_model/pos_model/pos_finetuning/step_size{}cm/gmm_{}%_pos_finetuning.pkl".format(int(step_size*100),i*10)
            get_gmm_finetuning(pkl_path,scene_num,step_size,root_emp,new_pkl_path)
    # file_path = f"scene_data/data_of_scene/data_of_scene{45}.txt"
    # root_current = generate_octree_from_txt(file_path)
    #
    # sim = compute_similarity(root_emp, root_current)
    # # sim2 = compute_layer_similarity(root_emp, root_current)
    # print("sim = :", sim)
    # # print("sim2 = :", sim2)
    # poential = compute_potential(root_history=root_emp, root_current=root_current)
    # print(len(poential))
    # for item in poential:
    #     print(item.index_array,'[', item.center[0], ',', item.center[1], ',', item.center[2], '],')
    # print(compute_total_force(poential, np.array([0.53115991, 0.09719978, 0.53595373], dtype=float) - center))
    # print(compute_total_force(poential, np.array([0.56462763, 0.12902873, 0.57918368], dtype=float) - center))
    # print(compute_total_force(poential, np.array([0.52360095, 0.02124487, 0.52451799], dtype=float) - center))
    # for item in root_current.points:
    #     print(item[0] + center[0], ' ', item[1] + center[1], ' ', item[2] + center[2])
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for item in root_current.points:
    #     print(item[0] + center[0], ' ', item[1] + center[1], ' ', item[2] + center[2])
