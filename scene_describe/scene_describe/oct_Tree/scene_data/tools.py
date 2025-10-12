import pandas as pd
import numpy as np


def read_3d_data(path):
    # 从excel表格中读取三维点，得到numpy数组
    excel_path = path
    sheet_name = 'Sheet1'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # 确定要读取的行数（不包括标题行）
    num_rows = df.shape[0]  # 这将获取包括标题行的总行数，如果你想要从第一行数据开始，需要减1（如果第一行是标题）

    # 由于要读取到x行，但这里的x实际上应该是行数，我们假设你想要读取所有数据行（除了可能的标题行）
    # 如果你有一个特定的x值（行数），你可以将num_rows替换为x（如果x是从0开始的索引，则需要注意+1以匹配实际的行数）
    # 例如：x = 10  # 如果你想读取前10行数据（从第0行开始计数，但通常Excel从第1行开始有数据）
    # num_rows = x + 1  # 如果包括标题行的话；如果不包括，则直接为x

    # 由于我们假设第一行是标题行，所以从第二行开始读取数据
    # 如果你想从第一行开始读取数据作为数据点，则不需要跳过任何行
    data_points = df.to_numpy()
    return data_points


def filter_scene_data(points, bound_up, bound_low, center):
    # 假设 points 是你的原始数组，每个元素都是一个形如 [x, y, z] 的三维坐标
    # 例如：points = np.array([[x1, y1, z1], [x2, y2, z2], ..., [xn, yn, zn]])
    # 把numpy数组形式的点进行过滤，得到在bound_up到bound_low的点，并且减去中心坐标，得到相对坐标
    x_low = bound_low[0]
    x_up = bound_up[0]
    y_low = bound_low[1]
    y_up = bound_up[1]
    z_low = bound_low[2]
    z_up = bound_up[2]

    # 定义条件
    x_condition = (points[:, 0] >= x_low) & (points[:, 0] <= x_up)
    y_condition = (points[:, 1] >= y_low) & (points[:, 1] <= y_up)
    z_condition = (points[:, 2] >= z_low) & (points[:, 2] <= z_up)

    # 筛选出同时满足所有条件的点
    filtered_points = points[x_condition & y_condition & z_condition]
    for item in filtered_points:
        item -= np.array(center)

    return filtered_points


def read_scene_data_in_txt(file_path):
    x_coords = []
    y_coords = []
    z_coords = []

    reading_x = False
    reading_y = False
    # 我们不需要reading_z变量，因为一旦开始读取z坐标，我们就知道x和y已经读取完毕

    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line == "下面是x坐标":
                reading_x = True
                reading_y = False
                continue
            elif line == "下面是y坐标":
                reading_x = False
                reading_y = True
                continue
            elif line == "下面是z坐标":
                # 在这里，我们可以假设x和y坐标已经被完全读取，因为文件是按顺序的
                # 因此，我们不需要再检查reading_x或reading_y，只需开始读取z坐标
                reading_y = False
                # 注意：我们不需要设置reading_z为True，因为一旦开始读取z，我们就知道是在这个块中
                continue

            if reading_x:
                x_coords.append(float(line))
            elif reading_y:
                y_coords.append(float(line))
            else:  # 这里else实际上只对应reading_z的情况（隐式地，因为文件是按顺序的）
                z_coords.append(float(line))

    # 检查所有坐标列表长度是否相同
    assert len(x_coords) == len(y_coords) == len(z_coords), "坐标列表长度不匹配"

    # 合并成n*3的NumPy数组
    coords_array = np.array([list(a) for a in zip(x_coords, y_coords, z_coords)])

    return coords_array
if __name__ == '__main__':
    x = read_scene_data_in_txt('./data_of_scene/data_of_scene0.txt')
    print(np.array(x))