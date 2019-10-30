import numpy as np
import matplotlib.pyplot as plt


def generate_data(m, n, begin=0, end=1):
    """
    生成数据
    :param m: 多项式阶数
    :param n: 训练集大小
    :param begin: 区间起点
    :param end: 区间终点
    :return: x, y, x = (n, m+1), y = (n, 1)
    """
    generate_x = np.arange(begin, end, (end - begin) / n)
    generate_y = np.sin(2 * np.pi * generate_x)
    noise = np.random.normal(0, 0.05, n)
    generate_y = generate_y + noise
    data_set = np.empty((n, m + 1))
    for i in range(n):
        data_set[i][0] = 1
        for j in range(1, m + 1):
            data_set[i][j] = data_set[i][j - 1] * generate_x[i]
    return data_set, generate_y.reshape(n, 1)


def normal_equation(x, y, _lambda=0):
    """
    正规方程即最小二乘法求解解析解，默认lambda=0无正则项
    :param x: 矩阵X
    :param y: 列向量y
    :param _lambda: 正则项系数，默认无正则项
    :return: 系数向量theta
    """
    m = x.shape[1] - 1
    # 正则项为 sigma(theta_i ** 2) i = 1 to m，与老师讲的二范数略有区别，这个正则项是参考吴恩达讲的的
    regular = np.eye(m + 1)
    regular[0][0] = 0
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x) + _lambda * regular), x.T), y)
    print(cost_function(theta, x, y))
    return theta


def cost_function(theta, x, y):
    """
    计算损失函数
    :param theta: 系数矩阵
    :param x: X矩阵
    :param y: y向量
    :return: 损失函数值
    """
    hx = np.dot(x, theta) - y
    cost = np.sum(np.power(hx, 2)) / (2 * y.size)
    return cost


def gradient_descent(x, y, alpha=0.05, epochs=500000, _lambda=0.0):
    """
    梯度下降法
    :param x: X矩阵
    :param y: y向量
    :param alpha: 学习率
    :param epochs: 迭代次数
    :param _lambda: 正则项系数，默认无正则项
    :return: 系数向量theta
    """
    # 正则项为 sigma(theta_i ** 2) i = 1 to m，与老师讲的二范数略有区别，这个正则项是参考吴恩达讲的的
    n, m = x.shape
    theta = np.zeros(m).reshape(m, 1)
    cost0 = 100000
    # y1 = np.array([])
    while epochs > 0:
        hx = (np.dot(x, theta) - y).T
        tmp_theta = (1 - _lambda * alpha / n) * theta - alpha / n * np.dot(hx, x).T
        theta = tmp_theta
        epochs -= 1
        cost1 = cost_function(theta, x, y)
        if cost1 > cost0:
            alpha /= 2
        # print(cost1-cost0)
        cost0 = cost1
    return theta


def conjugate_gradient(x_train, y_train, epochs=10000, eps=1e-10):
    """
    共轭梯度法
    :param x_train: X矩阵
    :param y_train: y向量
    :param epochs: 迭代次数
    :param eps: 精度
    :return: 系数向量theta
    """
    m, n = x_train.shape
    theta = (np.zeros(n)).reshape(n, 1)
    a = np.dot(x_train.T, x_train)
    b = np.dot(x_train.T, y_train)
    r0 = b - np.dot(a, theta)
    d0 = np.copy(r0)
    for i in range(epochs):
        alpha = np.sum(np.dot(r0.T, r0) / np.dot(np.dot(d0.T, a), d0))
        theta = theta + alpha * d0
        r1 = r0 - alpha * np.dot(a, d0)
        beta = np.sum(np.dot(r1.T, r1) / np.dot(r0.T, r0))
        if np.dot(r1.T, r1) < eps:
            return theta, i + 1
        d1 = r1 + beta * d0
        d0 = np.copy(d1)
        r0 = np.copy(r1)
        # print(cost_function(theta, x_train, y_train))
    return theta, epochs


def plot(theta, f, begin=0, end=1):
    func = np.poly1d(np.array(theta.T)[0][::-1])
    x1 = np.linspace(begin, end, 1000)
    y1 = func(x1)
    plt.plot(x1, y1, label=f)


def init(x_train, y_train, m, n):
    plt.title('m = ' + str(m) + ', n = ' + str(n))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_train[:, 1].T, y_train[:, 0].T, '.', label='original data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tm = 2
    tn = 10
    _x, _y = generate_data(tm, tn)
    # _theta = normal_equation(_x, _y, 0.01)
    # plot(_theta, 'normal_equation')
    _theta = gradient_descent(_x, _y, alpha=1)
    plot(_theta, 'gradient_descent')
    _theta, _epoch = conjugate_gradient(_x, _y)
    # plot(_theta, 'conjugate_gradient')
    init(_x, _y, tm, tn)
    plt.show()
