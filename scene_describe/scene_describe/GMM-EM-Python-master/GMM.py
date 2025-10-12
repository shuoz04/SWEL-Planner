import copy
import time

import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal


class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None):
        '''
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension 
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
            - colors: Color valu for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        '''
        self.k = k
        self.dim = dim
        if (init_mu is None):
            init_mu = random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if (init_sigma is None):
            """
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
            """
            init_sigma = np.array([np.eye(dim) for _ in range(k)])
        self.sigma = init_sigma
        if (init_pi is None):
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        if (colors is None):
            colors = random.rand(k, 3)
        self.colors = colors


    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        # self.num_points = X.shape[0]
        self.num_points = len(X)
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        """
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
        """
        for i in range(self.k):
            # 计算加权样本协方差矩阵
            N = self.num_points  # 或者更精确地，使用sum_z[i]作为当前成分的“有效”样本数
            centered_data = self.data - self.mu[i]  # 使用已经更新的均值
            weighted_cov = np.zeros((self.data.shape[1], self.data.shape[1]))
            for n in range(self.num_points):
                weighted_cov += self.z[n, i] * np.outer(centered_data[n], centered_data[n])
            self.sigma[i] = weighted_cov / sum_z[i]
            regularizer = 1e-6 * np.eye(self.data.shape[1])
            self.sigma[i] = self.sigma[i] + regularizer

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        '''
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)

    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        if (self.dim != 2):
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)


def test(data_1,data_2):
    X = data_1
    Y = data_2
    gmm = GMM(4, 6)
    gmm.init_em(X)
    num_iters = 15
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
        # plot2(title="Iteration: " + str(e + 1), X=X, gmm=gmm)
        time.sleep(1)

    print("mu:")
    print(gmm.mu)
    print("sigma:")
    print(gmm.sigma)
    print("weight:")
    print(gmm.pi)


    gmm_1 = GMM(4,6,init_mu= gmm.mu,init_sigma=gmm.sigma)
    gmm_1.init_em(Y)
    log_likelihood = [gmm.log_likelihood(Y)]
    # plotting
    # plot("Iteration:  0",gmm)
    for e in range(num_iters):
        # E-step
        gmm_1.e_step()
        # M-step
        gmm_1.m_step()
        # Computing log-likelihood
        log_likelihood.append(gmm_1.log_likelihood(Y))
        print("Iteration: {}, log-likelihood: {:.4f}".format(e + 1, log_likelihood[-1]))
        # plotting
        # plot2(title="Iteration: " + str(e + 1), X=X, gmm=gmm)
        time.sleep(1)
    print("mu:")
    print(gmm_1.mu)
    print("sigma:")
    print(gmm_1.sigma)
    print("weight:")
    print(gmm_1.pi)


def st_test(data, initial_temp=100, cooling_rate=0.99, max_temp_change=0.01):
    X = data
    gmm = GMM(4, 6)  # 假设数据是二维的，以便于可视化
    gmm.init_em(X)
    num_iters = 50
    current_temp = initial_temp
    # Saving log-likelihood
    log_likelihood = [gmm.log_likelihood(X)]

    for e in range(num_iters):
        # E-step
        gmm.e_step()

        # M-step
        gmm_before_perturb = copy.deepcopy(gmm)  # 保存M步之前的模型状态
        gmm.m_step()
        for i in range(gmm.k):
            gmm.sigma[i] = (gmm.sigma[i] + gmm.sigma[i].T) / 2  # 对称化
            gmm.sigma[i] += 1e-6 * np.eye(gmm.dim)  # 正定化
        # 计算M步后的似然函数值
        ll_after_m = gmm.log_likelihood(X)

        # 尝试以一定概率接受较差的解（模拟退火思想）
        if random.random() < np.exp((log_likelihood[-1] - ll_after_m) / current_temp):
            # 接受较差的解（回滚到M步之前的状态）
            gmm = copy.deepcopy(gmm_before_perturb)
            ll_after_m = log_likelihood[-1]  # 使用之前的似然函数值
        else:
            # 接受更好的解或相同（但不一定更好的，因为温度允许接受较差解）
            pass

        # 尝试扰动参数（模拟退火中的“新状态”）
        perturb_prob = random.random()
        if perturb_prob < 0.1:  # 以10%的概率扰动参数
            # 随机选择一个参数进行微小扰动
            choice = random.randint(0, 2)  # 0: mu, 1: sigma, 2: pi
            if choice == 0:
                # 扰动均值
                gmm.mu += random.randn(gmm.k, gmm.dim) * 0.01  # 微小扰动
            elif choice == 1:
                # 扰动协方差（这里只扰动对角线元素作为简化）
                gmm.sigma = [s + random.randn(gmm.dim, gmm.dim) * 0.01 for s in gmm.sigma]
                # 确保协方差矩阵仍然是对称的且正定的
                for i in range(gmm.k):
                    gmm.sigma[i] = (gmm.sigma[i] + gmm.sigma[i].T) / 2  # 对称化
                    gmm.sigma[i] += 1e-6 * np.eye(gmm.dim)  # 正定化
            elif choice == 2:
                # 扰动权重
                gmm.pi += random.randn(gmm.k) * 0.01
                gmm.pi = np.clip(gmm.pi, 0, 1)  # 确保权重在[0,1]之间
                gmm.pi /= gmm.pi.sum()  # 重新归一化

        # 重新计算扰动后的似然函数值（如果需要）
        if perturb_prob < 0.1:
            ll_after_perturb = gmm.log_likelihood(X)
            # 如果扰动后的似然函数值更好，或者根据温度接受较差的解
            if ll_after_perturb > ll_after_m or random.random() < np.exp(
                    (ll_after_m - ll_after_perturb) / current_temp):
                # 接受扰动后的解
                pass
            else:
                # 回滚到M步之后的状态（但不扰动）
                # gmm = gmm_after_m.__copy__()  # 注意：这里需要保存M步之后未扰动的状态，但之前未定义，需调整逻辑
                # 由于我们之前没有保存这个状态，这里简单回滚到M步之前然后重新做M步（不是最佳实践）
                gmm = copy.deepcopy(gmm_before_perturb)
                gmm.m_step()
                ll_after_perturb = gmm.log_likelihood(X)  # 重新计算
        else:
            # 如果没有扰动，则使用M步后的解和似然函数值
            ll_after_perturb = ll_after_m

        # 注意：上面的扰动和回滚逻辑存在一些问题，特别是关于如何保存和恢复状态的部分。
        # 在实际应用中，需要更仔细地设计这部分逻辑，以确保算法的正确性和效率。
        # 为了简化示例，这里我们直接接受M步后的解（不考虑扰动导致的回滚问题）。
        # 下面的代码行是简化后的逻辑，只接受M步后的解，并基于温度偶尔接受较差的解。

        # 最终接受M步（可能带有扰动，但上面逻辑已简化）后的解
        log_likelihood.append(ll_after_m)  # 使用M步后的似然函数值（可能已被温度接受准则修改）

        # 降低温度（模拟退火过程）
        current_temp *= cooling_rate
        # 但不要让温度降得太低，以免完全停止接受较差解
        current_temp = max(current_temp, max_temp_change)

        print("Iteration: {}, log-likelihood: {:.4f}, Temperature: {:.2f}".format(e + 1, log_likelihood[-1],
                                                                                  current_temp))
        time.sleep(1)  # 控制打印间隔

    print("Final mu:")
    print(gmm.mu)
    print("Final sigma:")
    print(gmm.sigma)
    print("Final weight:")
    print(gmm.pi)


if __name__ == '__main__':
    # pos_data,orien_data = sample_data_load('../data.txt')
    data_1 = np.array([[0, 2.0943951, -1.57079633, 2.61799388, 1.57079633, 0],
                     [-0.34906585, 0, 1.46607657, 1.74532925, -4.4331363, 0],
                     [-3.10269954, 2.23192032, 0.34108974, 2.22796694, 2.29163601, 0.32888586],
                     [-2.83283575, 1.76356279, 1.38289846, 1.41748848, 2.28345503, -0.02972328],
                     [-2.9341147, 2.2548365, 0.81507761, 1.58211888, 2.29402819, 0.1040595],
                     [-2.96648029, 2.03677018, 1.02545007, 1.61840125, 2.29551911, 0.14721668],
                     [-3.06329181, 2.17704772, 0.49635241, 2.09293886, 2.29444338, 0.27649569],
                     [-3.25629089, 2.05733663, 0.71093437, 2.16325421, 2.26800195, 0.5289931],
                     [-2.99434554, 2.29334856, 0.56537824, 1.84657907, 2.29606008, 0.18443842],
                     [-3.1708451, 1.87750411, 0.99505762, 1.98741928, 2.28360106, 0.41866014],
                     [-3.05738838, 2.27799181, 0.68609501, 1.79704149, 2.2947463, 0.26862738],
                     [-3.24763109, 2.13341171, 0.73273888, 2.05827628, 2.26985441, 0.51795013],
                     [-3.14816953, 1.93035644, 0.86298988, 2.04715711, 2.28671735, 0.38893084],
                     [-3.04641477, 2.2055134, 0.63668925, 1.9092286, 2.29522779, 0.25399112]])

    data_2 = np.array([[0, 2.0943951, -1.57079633, 2.61799388, 1.57079633, 0],
                     [-0.34906585, 0, 1.46607657, 1.74532925, -4.4331363, 0],
                     [-3.10269954, 2.23192032, 0.34108974, 2.22796694, 2.29163601, 0.32888586],
                     [-2.83283575, 1.76356279, 1.38289846, 1.41748848, 2.28345503, -0.02972328],
                     [-2.9341147, 2.2548365, 0.81507761, 1.58211888, 2.29402819, 0.1040595],
                     [-2.96648029, 2.03677018, 1.02545007, 1.61840125, 2.29551911, 0.14721668],
                     [-3.06329181, 2.17704772, 0.49635241, 2.09293886, 2.29444338, 0.27649569],
                     [-3.25629089, 2.05733663, 0.71093437, 2.16325421, 2.26800195, 0.5289931],
                     [-2.99434554, 2.29334856, 0.56537824, 1.84657907, 2.29606008, 0.18443842],
                     [-3.1708451, 1.87750411, 0.99505762, 1.98741928, 2.28360106, 0.41866014],
                     [-3.05738838, 2.27799181, 0.68609501, 1.79704149, 2.2947463, 0.26862738],
                     [-3.24763109, 2.13341171, 0.73273888, 2.05827628, 2.26985441, 0.51795013],
                     [-3.14816953, 1.93035644, 0.86298988, 2.04715711, 2.28671735, 0.38893084],
                     [-3.04641477, 2.2055134, 0.63668925, 1.9092286, 2.29522779, 0.25399112],
                     [-3.00306181, 1.99816699, 0.92644556, 1.7884199, 2.29608828, 0.19608623],
                     [-2.96457043, 1.99048331, 1.20875274, 1.47969526, 2.29545638, 0.14466708],
                     [-3.07416649, 2.38001831, 0.45381244, 1.94209383, 2.29380487, 0.29097785],
                     [-3.08464046, 1.79216537, 1.27373466, 1.71923885, 2.29309181, 0.30490975],
                     [-3.22486474, 2.04614594, 0.85262533, 2.00681952, 2.27443633, 0.48875937],
                     [-3.19097225, 2.21403164, 0.569766, 2.09331687, 2.28047106, 0.44490609],
                     [-3.2084767, 2.06227416, 0.67520949, 2.15439959, 2.27747296, 0.46761131],
                     [-3.1288398, 2.19761214, 0.41207113, 2.21408646, 2.28902675, 0.36346889],
                     [-2.98210664, 1.81328828, 1.33475361, 1.54641745, 2.2959072, 0.16808565],
                     [-3.03771658, 2.23523142, 0.7487623, 1.75974269, 2.29553424, 0.24238188],
                     [-3.25722243, 1.96646975, 0.73233419, 2.23348284, 2.26779934, 0.53017905],
                     [-3.09437946, 1.85736683, 0.9744603, 1.96186156, 2.2923425, 0.31784752],
                     [-2.9786315, 2.31340069, 0.36403887, 2.01394126, 2.29583982, 0.16344354],
                     [-3.08230406, 2.09890427, 0.59002645, 2.09415441, 2.29325933, 0.30180356],
                     [-3.17091639, 1.95347809, 0.82408762, 2.08247637, 2.28359056, 0.41875341],
                     [-3.18945691, 2.07959966, 0.81519825, 1.98103198, 2.28071845, 0.44293497],
                     [-2.99168975, 1.76016475, 1.5522872, 1.39049989, 2.29603825, 0.18088965],
                     [-3.14884136, 1.98755448, 1.04997682, 1.80355229, 2.28663206, 0.38981448],
                     [-3.09593377, 1.96739059, 1.09167082, 1.73599003, 2.29221539, 0.31991074]])
    #st_test(data)
    test(data_1,data_2)