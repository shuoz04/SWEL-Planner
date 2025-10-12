import pandas as pd
import logging
import numpy as np
from typing import Optional, Tuple, List

def read_3d_data(path):
    # 从excel表格中读取三维点，得到numpy数组
    excel_path = path
    sheet_name = 'Sheet1'
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # 确定要读取的行数（不包括标题行）
    num_rows = df.shape[0]  # 这将获取包括标题行的总行数，如果你想要从第一行数据开始，需要减1（如果第一行是标题）

    # 由于你说要读取到x行，但这里的x实际上应该是行数，我们假设你想要读取所有数据行（除了可能的标题行）
    # 如果你有一个特定的x值（行数），你可以将num_rows替换为x（如果x是从0开始的索引，则需要注意+1以匹配实际的行数）
    # 例如：x = 10  # 如果你想读取前10行数据（从第0行开始计数，但通常Excel从第1行开始有数据）
    # num_rows = x + 1  # 如果包括标题行的话；如果不包括，则直接为x

    # 由于我们假设第一行是标题行，所以从第二行开始读取数据
    # 如果你想从第一行开始读取数据作为数据点，则不需要跳过任何行
    data_points = df.to_numpy()
    return data_points


def filter_scene_data(
        points: np.ndarray,
        bound_low: Tuple[float, float, float],
        bound_up: Tuple[float, float, float],
        center: Optional[Tuple[float, float, float]] = None,
        inplace: bool = False
) -> np.ndarray:
    """
    Filters 3D points within specified bounds and optionally centers them.

    Args:
        points: Nx3 numpy array of 3D coordinates
        bound_low: Tuple of (min_x, min_y, min_z) defining lower bounds
        bound_up: Tuple of (max_x, max_y, max_z) defining upper bounds
        center: Optional center point for coordinate transformation. If None, no centering performed.
        inplace: If True, modifies the input array (when safe). False by default.

    Returns:
        Filtered (and optionally centered) points as a new array

    Raises:
        ValueError: Invalid bounds (max < min) or shape mismatch
    """
    # Input validation
    if points.shape[1] != 3:
        raise ValueError("Points must be an Nx3 array")
    if not all(u > l for l, u in zip(bound_low, bound_up)):
        raise ValueError("Upper bounds must be greater than lower bounds for all dimensions")

    # Convert bounds to numpy arrays for vectorized operations
    bound_low = np.asarray(bound_low)
    bound_up = np.asarray(bound_up)

    # Create mask for points within bounds
    in_bounds_mask = (
            (points[:, 0] >= bound_low[0]) & (points[:, 0] <= bound_up[0]) &
            (points[:, 1] >= bound_low[1]) & (points[:, 1] <= bound_up[1]) &
            (points[:, 2] >= bound_low[2]) & (points[:, 2] <= bound_up[2])
    )

    # Select points (create copy if not operating inplace)
    if inplace and np.all(in_bounds_mask):
        filtered = points  # Use entire array if all points pass filter
    else:
        filtered = points[in_bounds_mask].copy()  # Ensure we get a new array

    # Center coordinates if requested
    if center is not None:
        center = np.asarray(center)
        if center.size != 3:
            raise ValueError("Center must be a 3-element coordinate")
        filtered -= center  # Vectorized subtraction

    return filtered


def read_data_from_txt(file_path: str) -> np.ndarray:
    """
    从结构化文本文件读取三维坐标数据

    文件格式要求：
    - 以"下面是x坐标"、"下面是y坐标"、"下面是z坐标"作为分隔标记
    - 每个标记后跟随对应坐标的数值列表
    - 文件编码应为UTF-8

    Args:
        file_path: 文本文件路径

    Returns:
        np.ndarray: N×3的numpy数组，每行表示一个三维点

    Raises:
        ValueError: 如果坐标数量不匹配或文件格式错误
        FileNotFoundError: 如果文件不存在
        UnicodeDecodeError: 如果编码不是UTF-8
    """
    # 初始化坐标存储
    coord_buffers: Tuple[List[float], List[float], List[float]] = ([], [], [])
    current_section = None  # 跟踪当前读取的坐标类型
    expected_sections = ["x", "y", "z"]
    section_markers = {f"下面是{axis}坐标": axis for axis in expected_sections}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                # 检查是否是分段标记
                if line in section_markers:
                    current_section = section_markers[line]
                    continue

                # 处理数据行
                try:
                    value = float(line)
                except ValueError as e:
                    raise ValueError(
                        f"Line {line_num}: 无效数值 '{line}' (当前分段: {current_section}坐标)"
                    ) from e

                if current_section == "x":
                    coord_buffers[0].append(value)
                elif current_section == "y":
                    coord_buffers[1].append(value)
                elif current_section == "z":
                    coord_buffers[2].append(value)
                else:
                    raise ValueError(
                        f"Line {line_num}: 数据不在任何已知坐标分段中"
                    )

    except FileNotFoundError:
        logging.error(f"文件未找到: {file_path}")
        raise
    except UnicodeDecodeError:
        logging.error(f"文件编码不是UTF-8: {file_path}")
        raise

    # 验证数据完整性
    x_len, y_len, z_len = map(len, coord_buffers)
    if x_len != y_len or y_len != z_len:
        error_msg = f"坐标数量不匹配 (x:{x_len}, y:{y_len}, z:{z_len})"
        logging.error(error_msg)
        raise ValueError(error_msg)

    if x_len == 0:
        logging.warning("警告: 读取到空数据集")

    # 更高效的数组构建方式
    return np.column_stack(coord_buffers)


if __name__ == '__main__':
    # path = './experiment/data/scene_data/20%.txt'
    # points = read_data_from_txt(path)
    # low = [0.46, 0, 0.48]
    # up = [0.7, 0.2, 0.68]
    # center = np.array([0.58, 0.08, 0.57])
    # scene = filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    # for item in scene:
    #     print(item[0]+0.417,' ',item[1]+0.0833,' ',item[2]+0.583)
    # (0.25+1/6,1/12 , 0.25+1/3)
    # excel_path = './scene_data/baseline_scene.xlsx'
    # points = read_3d_data(excel_path)
    # path = "./experiment/data/scene_data/20%.txt"
    path = "./scene_data/data_of_scene/data_of_scene45.txt"
    points = read_data_from_txt(path)
    low = [0.46-0.16, 0-0.16, 0.48-0.16]
    up = [0.7+0.16, 0.2+0.16, 0.68+0.16]
    center = np.array([0.58, 0.08, 0.57])
    scene_1 = filter_scene_data(points=points, bound_low=low, bound_up=up, center=center)
    for item in scene_1:
        # print(item[0]+0.397,' ',item[1]+0.0833,' ',item[2]+0.583)
        print(item[0]+0.397,' ',item[1]+0.0833,' ',item[2]+0.583)
