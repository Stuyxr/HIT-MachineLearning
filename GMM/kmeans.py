import numpy as np
import random
import matplotlib.pyplot as plt


color = ['r', 'g', 'b', 'c']


def generate_x(mean, var, size):
    x = np.array([])
    for i in range(len(mean)):
        tmp = np.random.normal(loc=mean[i], scale=var, size=size)
        x = np.append(x, tmp)
    x = x.reshape(-1, size).T
    return x


def generate_data():
    """
    生成数据
    :return: None
    """
    mean1 = [-6, 6]
    mean2 = [6, -6]
    mean3 = [3, 3]
    mean4 = [-3, -3]
    var = 1
    size1 = size2 = size3 = size4 = 100
    x1 = generate_x(mean1, var, size1)
    x2 = generate_x(mean2, var, size2)
    x3 = generate_x(mean3, var, size3)
    x4 = generate_x(mean4, var, size4)
    x1 = np.row_stack((x1, x2))
    x1 = np.row_stack((x1, x3))
    x1 = np.row_stack((x1, x4))
    np.random.shuffle(x1)
    return x1


def random_center(x):
    """
    随机中心点
    :param x: 样本
    :return: 中心点
    """
    index = random.sample(list(range(len(x))), 4)
    center = np.array([x[i, :] for i in index])
    return center.reshape(4, 2)


def get_distance(x, y):
    """
    计算两个向量x y的距离
    :param x: 向量x
    :param y: 向量y
    :return: 距离
    """
    return np.sqrt(np.sum(np.square(x - y)))


def calculate_distance(x, center):
    x_size = np.shape(x)[0]
    k = np.shape(center)[0]
    dis = np.zeros((x_size, k))
    for i in range(x_size):
        for j in range(k):
            dis[i, j] = get_distance(x[i, :], center[j, :])
    return dis


def get_means(data):
    tot_x = tot_y = 0
    for i in range(len(data)):
        tot_x += data[i][0]
        tot_y += data[i][1]
    return tot_x / len(data), tot_y / len(data)


def kmeans(x):
    """
    k means算法
    :param x: 样本
    :return: 中心点
    """
    center = random_center(x)
    size = np.shape(x)[0]
    k = np.shape(center)[0]
    for i in range(1000):
        dis = calculate_distance(x, center)
        y = np.argmin(dis, axis=1)
        clusters = [[] for j in range(k)]
        for j in range(size):
            clusters[y[j]].append(x[j, :].tolist())
        new_center = np.array([])
        for j in range(k):
            new_center = np.append(new_center, get_means(clusters[j]))
        new_center = new_center.reshape(k, -1)
        if (new_center == center).all():
            print('已收敛，迭代次数{}'.format(i + 1))
            return y
        center = new_center
    return y


def plot(x, y):
    for i in range(y.size):
        plt.plot(x[i, 0], x[i, 1], '.', color=color[y[i]])
    plt.show()


if __name__ == '__main__':
    x = generate_data()
    y = kmeans(x)
    plot(x, y)
