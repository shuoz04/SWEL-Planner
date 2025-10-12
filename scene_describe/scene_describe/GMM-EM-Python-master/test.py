import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
from scipy.stats import multivariate_normal
# import GMM
from GMM import GMM
import time


def sample_data_load(path):
    file_path = path
    # 假设txt文件的路径为'coordinates.txt'
    # 初始化一个空列表来存储所有的三维点坐标
    all_points = []

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 确保数据是3的倍数，即每三个浮点数组成一个三维点
        if len(lines) % 3 != 0:
            raise ValueError("文件中的浮点数据行数不是3的倍数，无法组成完整的三维点坐标。")

        # 每次取三行数据，组成一个三维点坐标，并添加到all_points列表中
        for i in range(0, len(lines), 3):
            # 将三行数据转换为浮点数，并组成一个numpy数组
            point = np.array([float(lines[i].strip()), float(lines[i + 1].strip()), float(lines[i + 2].strip())])
            all_points.append(point)

    # 将所有的三维点坐标列表转换为numpy数组
    all_points_array = np.array(all_points)
    pos_data = []
    orien_data = []
    for i in range(all_points_array.shape[0]):
        if i % 2 == 0:
            pos_data.append(all_points_array[i])
        else:
            orien_data.append(all_points_array[i])
    pos_data = np.array(pos_data)
    orien_data = np.array(orien_data)
    return pos_data, orien_data


def gen_data(k=3, dim=2, points_per_cluster=200, lim=[-10, 10]):
    '''
    Generates data from a random mixture of Gaussians in a given range.
    Will also plot the points in case of 2D.
    input:
        - k: Number of Gaussian clusters
        - dim: Dimension of generated points
        - points_per_cluster: Number of points to be generated for each cluster
        - lim: Range of mean values
    output:
        - X: Generated points (points_per_cluster*k, dim)
    '''
    x = []
    mean = random.rand(k, dim) * (lim[1] - lim[0]) + lim[0]
    for i in range(k):
        cov = random.rand(dim, dim + 10)
        cov = np.matmul(cov, cov.T)
        _x = np.random.multivariate_normal(mean[i], cov, points_per_cluster)
        x += list(_x)
    x = np.array(x)
    if (dim == 2):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4)
        ax.autoscale(enable=True)
    return x


def plot(title, gmm, X):
    '''
    Draw the data points and the fitted mixture model.
    input:
        - title: title of plot and name with which it will be saved.
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.4)
    ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c=gmm.colors)
    gmm.draw(ax, lw=3)
    ax.set_xlim((-12, 12))
    ax.set_ylim((-12, 12))

    plt.title(title)
    plt.savefig(title.replace(':', '_'))
    plt.show()
    plt.clf()


def plot2(title, X, gmm):
    '''
    Draw the data points and the fitted mixture model in 3D.
    input:
        - title: title of plot and name with which it will be saved.
        - X: 3D data points.
        - gmm: the fitted Gaussian Mixture Model object (assuming it has a 3D draw method).
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axes

    # Assuming X is a (N, 3) array where each row is (x, y, z)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, alpha=0.4)  # Plot data points

    # Calculate the min and max values for the x, y, and z axes
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()

    # Apply some padding to the limits for better visualization (optional)
    padding = 0.1  # You can adjust this padding value as needed
    x_min, x_max = x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min)
    y_min, y_max = y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min)
    z_min, z_max = z_min - padding * (z_max - z_min), z_max + padding * (z_max - z_min)

    # Set the 3D limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # For the GMM means, assuming they are also 3D and gmm.draw_3d is a custom method
    # If not, you'll need to manually plot them or adjust the data to be 3D
    if hasattr(gmm, 'draw_3d'):
        gmm.draw_3d(ax, lw=3)  # Custom method to draw GMM in 3D
    else:
        # Fallback: Plot means as points
        ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], gmm.mu[:, 2], c=gmm.colors, s=100, alpha=1.0)

    filename = title.replace(':', '_') + '.png'  # 通常我们会添加一个文件扩展名，如 '.png'
    save_path = os.path.join(os.getcwd(), 'results', filename)
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to prevent memory leaks


def plot1(title, X, gmm):
    '''
        Draw the data points and the fitted mixture model in 3D.
        input:
            - title: title of plot and name with which it will be saved.
            - X: 3D data points.
            - gmm: the fitted Gaussian Mixture Model object (assuming it has a 3D draw method).
        '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axes

    # Assuming X is a (N, 3) array where each row is (x, y, z)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=3, alpha=0.4)  # Plot data points

    # For the GMM means, assuming they are also 3D and gmm.draw_3d is a custom method
    # If not, you'll need to manually plot them or adjust the data to be 3D
    if hasattr(gmm, 'draw_3d'):
        gmm.draw_3d(ax, lw=3)  # Custom method to draw GMM in 3D
    else:
        # Fallback: Plot means as points
        ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], gmm.mu[:, 2], c=gmm.colors, s=100, alpha=1.0)

        # Set 3D limits (optional, adjust as needed)
    ax.set_xlim((0, 0.3))
    ax.set_ylim((0, 0.7))
    ax.set_zlim((0.3, 0.8))
    filename = title.replace(':', '_') + '.png'  # 通常我们会添加一个文件扩展名，如 '.png'
    save_path = os.path.join(os.getcwd(), 'logs/k4d3/result1_6906.7232/results', filename)
    plt.title(title)
    plt.savefig(save_path)

    # Note: This function assumes that gmm has a 3D draw method or that gmm.mu is 3D.


# If not, you need to adjust the data or implement the 3D drawing functionality.
def dataload():
    # 读取Excel文件
    file_path = './datasets/points.xlsx'  # Excel文件路径
    sheet_name = 'Sheet1'  # Excel工作表名称

    # 使用pandas读取数据，假设数据在第一列，我们可以使用列名（如果设置了）或列的索引（0表示第一列）
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[0])

    # 将pandas Series转换为列表
    data_list = data.iloc[:, 0].tolist()
    x = [data_list[i:i + 3] for i in range(0, len(data_list), 3)]
    # print(x)
    X = np.array(x)
    return X


def test(data, gmm: GMM, iteration_num):
    X = data
    gmm.init_em(X)
    num_iters = iteration_num
    # Saving log-likelihood
    log_likelihood = [gmm.log_likelihood(X)]
    # plotting
    # plot("Iteration:  0",gmm)
    for e in range(num_iters):
        # E-step
        gmm.e_step()
        # M-step
        gmm.m_step()
        # Computing log-likelihood
        log_likelihood.append(gmm.log_likelihood(X))
        print("Iteration: {}, log-likelihood: {:.4f}".format(e + 1, log_likelihood[-1]))
        # plotting
        plot2(title="Iteration: " + str(e + 1), X=X, gmm=gmm)
        time.sleep(1)
    print("mu:")
    print(gmm.mu)
    print("sigma:")
    print(gmm.sigma)
    print("weight:")
    print(gmm.pi)


def read_data_from_txt(path):
    filename = path
    coordinates = []

    with open(filename, 'r') as file:
        for line in file:
            # 去除行尾的换行符，然后去除方括号（假设方括号是每行的第一个和最后一个字符）
            line = line.strip()[1:-1]
            # 按空格分割字符串，并将结果转换为浮点数数组
            point = np.fromstring(line, sep=' ')
            # 将numpy数组添加到列表中
            coordinates.append(point)

    # 如果需要，将整个列表转换为一个大numpy数组
    coordinates_array = np.array(coordinates)

    # 打印结果以验证
    print(coordinates_array)
    return coordinates_array


def remove_outliers_by_boxplot(points, iqr_threshold=1.2):
    # 去除数据中的离群点，参数1.4可调
    """
    Remove 3D points that are outliers based on box plot analysis of each dimension.

    Parameters:
    points (numpy.ndarray): A Nx3 array of 3D points.
    iqr_threshold (float): The threshold for identifying outliers based on the interquartile range.
                           Default is 1.5, which means points outside 1.5*IQR from the Q3 or Q1 are considered outliers.

    Returns:
    numpy.ndarray: A Mx3 array of filtered 3D points, where M <= N.
    """
    # Extract each dimension
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute the 1st quartile (Q1) and 3rd quartile (Q3)
    Q1_x, Q3_x = np.percentile(x, [25, 75])
    Q1_y, Q3_y = np.percentile(y, [25, 75])
    Q1_z, Q3_z = np.percentile(z, [25, 75])

    # Compute the interquartile range (IQR)
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y
    IQR_z = Q3_z - Q1_z

    # Define the lower and upper bounds for non-outlier points
    lower_bound_x, upper_bound_x = Q1_x - iqr_threshold * IQR_x, Q3_x + iqr_threshold * IQR_x
    lower_bound_y, upper_bound_y = Q1_y - iqr_threshold * IQR_y, Q3_y + iqr_threshold * IQR_y
    lower_bound_z, upper_bound_z = Q1_z - iqr_threshold * IQR_z, Q3_z + iqr_threshold * IQR_z

    # Create boolean masks for non-outlier points
    mask_x = (x >= lower_bound_x) & (x <= upper_bound_x)
    mask_y = (y >= lower_bound_y) & (y <= upper_bound_y)
    mask_z = (z >= lower_bound_z) & (z <= upper_bound_z)

    # Combine the masks to get the final mask of non-outlier points
    mask = mask_x & mask_y & mask_z

    # Apply the mask to filter out the outliers
    filtered_points = points[mask]

    return filtered_points


if __name__ == '__main__':
    # pos_data,orien_data = sample_data_load('../data.txt')
    data = read_data_from_txt('./datasets/3D_sample_experience_data.txt')
    points = data  # 100 random points in 3D space
    # Add some outliers (points far away from the others)
    outliers = np.array([[0, 0, 0], [1, 1, 1], [0.9, 0.95, 0.99], [0.1, 0.15, 0.2]])
    points = np.vstack((points, outliers))
    # Remove outliers using box plot analysis
    filtered_points = remove_outliers_by_boxplot(points)

    k = 4
    dim = 3
    gmm = GMM(k, dim)
    test(filtered_points, gmm, 100)
    # dataload()
    # plt.plot(log_likelihood[1:], marker='.')
    """
    for i in range(1, len(log_likelihood)):
        plt.title("log-likelihood for iteration: " + str(i))
        plt.plot(log_likelihood[1:1 + i], marker='.')
        axes = plt.axes()
        axes.set_ylim([min(log_likelihood[1:]) - 50, max(log_likelihood[1:]) + 50])
        axes.set_xlim([-2, 32])
        plt.savefig("ll_" + str(i))
        plt.clf()
    """
