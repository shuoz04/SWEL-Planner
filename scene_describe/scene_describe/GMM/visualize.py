import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import data_tools as tl

def quat_rotate_vector(q, v):
    """ 使用四元数 q 旋转向量 v """
    q_rot = R.from_quat(q)
    return q_rot.apply(v)


def plot_quaternions_with_weights(quaternions, weights):
    # 原始向量 (0,0,1)
    vector = np.array([0, 0, 1])

    # 计算每个四元数旋转后的向量
    rotated_vectors = np.array([quat_rotate_vector(q, vector) for q in quaternions])

    # 计算平均四元数
    avg_quat = np.mean(quaternions, axis=0)
    avg_rotated_vector = quat_rotate_vector(avg_quat, vector)

    # 创建图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个四元数旋转后的向量，颜色根据权重变化
    for i in range(len(quaternions)):
        weight = weights[i]
        color = plt.cm.summer((weight - 0.5) / 0.5)  # 从浅绿到深绿

        ax.quiver(0, 0, 0, rotated_vectors[i, 0], rotated_vectors[i, 1], rotated_vectors[i, 2],
                  color=color, length=1.0, normalize=True)

    # 绘制参考四元数旋转后的向量（黑色）
    ax.quiver(0, 0, 0, avg_rotated_vector[0], avg_rotated_vector[1], avg_rotated_vector[2],
              color='k', length=1.0, normalize=True, label='参考四元数')

    # 设置图形标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 去掉网格和坐标轴
    ax.grid(False)  # 去掉网格
    ax.set_axis_off()  # 去掉坐标轴

    # 设置视角，调整使得箭头都能显示
    ax.view_init(elev=30, azim=45)  # 视角调整

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()


# 示例：一些四元数和权重数据
path = "./dataset/data.txt"
data = tl.data_read(path)
data_ = tl.data_process(data)
weights,angle = tl.getWeightsforData(data)
quaternions = data_

  # 权重，从0.5到1

# 可视化四元数旋转后的向量
plot_quaternions_with_weights(quaternions, weights)


