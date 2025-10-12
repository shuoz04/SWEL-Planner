
from scene_data import tools
import oct_tree_simple as oct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_octree_from_txt(path):
    # 从一个txt文档里面读取点，然后过滤到指定栅格，然后计算相对坐标，最好生成八叉树，返回根节点
    points = tools.read_scene_data_in_txt(path)
    low = [0.46, 0, 0.48]
    up = [0.7, 0.2, 0.68]
    center = np.array([0.58, 0.08, 0.57])
    scene = tools.filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    root = oct.OctreeNode(0, depth=1, parent=None)
    for item in scene:
        oct.insert_node(root, item)
    return root


if __name__ == '__main__':
    # 得到示例数据的八叉树根节点
    excel_path = './scene_data/data_of_scene1.xlsx'
    points = tools.read_3d_data(excel_path)
    low = [0.46, 0, 0.48]
    up = [0.7, 0.2, 0.68]
    center = np.array([0.58, 0.08, 0.57])
    scene_1 = tools.filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    # 生成八叉树
    root_emp = oct.OctreeNode(0, depth=1, parent=None)
    for item in scene_1:
        oct.insert_node(root_emp, item)
    # 下面得到其他txt中的测试八叉树数据
    roots = []  # 存储根节点
    num_of_scene = 1000  # 场景数据数量
    sims = []  # 存储所有相似度
    num_of_strong = 0
    num_of_mid = 0
    for i in range(1000):
        file_path = f"scene_data/data_of_scene/data_of_scene{i}.txt"
        roots.append(generate_octree_from_txt(file_path))
    for item in roots:
        sims.append(oct.compute_sim(root_emp, item))
    for i in range(num_of_scene):
        if sims[i] > 0.7:
            num_of_strong += 1
            print(i,sims[i])
        if (sims[i] < 0.7) and (sims[i] > 0.5):
            num_of_mid += 1
    print("强相似占比：", num_of_strong / num_of_scene)
    print("若相似占比：", num_of_mid / num_of_scene)
    num_of_strong = 0
    num_of_mid = 0
    sim_max = 0
    sims = np.array(sims)
    for i in range(10):
        for j in range(100*i,100*i + 100):
            if oct.compute_sim(root_emp, roots[j]) > 0.8:
                if oct.compute_sim(root_emp, roots[j]) > sim_max:
                    sim_max = oct.compute_sim(root_emp, roots[j])
                num_of_strong += 1
            if oct.compute_sim(root_emp, roots[j]) > 0.5 and oct.compute_sim(root_emp, roots[j])<0.8:
                num_of_mid += 1
        print("{}到{}：".format(0,100*i +100),num_of_strong / (100*i +100)," ",num_of_mid / (100*i+100),"最大相似度：",sim_max)
    print("------------------------------------------------------------------------------")
    print(sims)
    """
    max_value = np.max(np.array(sims))
    max_index = np.argmax(np.array(sims)
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
