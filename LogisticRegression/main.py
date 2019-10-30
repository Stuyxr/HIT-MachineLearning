import numpy as np
import matplotlib.pyplot as plt


def generate_data(loc1, scale1, size1, loc2, scale2, size2):
    """
    生成数据
    :param loc1: 类别1的均值
    :param scale1: 类别1的方差
    :param size1: 类别1数据个数
    :param loc2: 类别2的均值
    :param scale2: 类别2的方差
    :param size2: 类别2数据个数
    :return: 数据集
    """
    x1 = np.empty((size1, loc1.size))
    x2 = np.empty((size2, loc2.size))
    y1 = np.ones(size1)
    y2 = np.zeros(size2)
    for i in range(size1):
        for j in range(loc1.size):
            x1[i][j] = np.random.normal(loc1[j], scale1[j], 1)
    for i in range(size2):
        for j in range(loc2.size):
            x2[i][j] = np.random.normal(loc2[j], scale2[j], 1)
    if loc1.size == 2:
        plt.plot(x1[:, 0:1], x1[:, 1:], '.', color='red')
        plt.plot(x2[:, 0:1], x2[:, 1:], '.', color='blue')
    x1 = np.column_stack((x1, y1))
    x2 = np.column_stack((x2, y2))
    x1 = np.row_stack((x1, x2))
    np.random.shuffle(x1)
    return x1[:, :-1], x1[:, -1:]


def sigmoid(x):
    """
    sigmoid函数
    :param x: x
    :return: sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def cost_function(theta, x, y):
    """
    J(theta) =  -1/m * sum(y_i * log(sigmoid(x_i * theta)) + (1 - y_i) * log(1 - sigmoid(x_i * theta)))
    :param theta: 参数向量theta
    :param x: x
    :param y: y
    :return: 代价函数的值
    """
    cost = np.sum(y.T * np.log(sigmoid(np.dot(theta, x))) + (1 - y).T * np.log(1 - sigmoid(np.dot(theta, x))))
    cost = -1 / y.size * cost
    return cost


def gradient_descent(x, y, alpha=0.05, epochs=100000, _lambda=0.0):
    """
    梯度下降法
    :param x: X矩阵
    :param y: y向量
    :param alpha: 学习率
    :param epochs: 迭代次数
    :param _lambda: 正则项系数，默认无正则项
    :return: 系数向量theta
    """
    x = np.column_stack((np.ones(y.size).T, x))
    n, m = x.shape
    theta = np.zeros(m).reshape(m, 1)
    # y1 = np.array([])
    while epochs > 0:
        hx = (sigmoid(np.dot(x, theta)) - y).T
        theta = (1 - _lambda * alpha / n) * theta - alpha / n * np.dot(hx, x).T
        epochs -= 1
    return theta


def get_accuracy(theta, x, y):
    """
    准确率
    :param theta:
    :param x:
    :param y:
    :return:
    """
    x = np.column_stack((np.ones(y.size).T, x))
    predict_y = sigmoid(np.dot(x, theta))
    predict_y = predict_y.flatten()
    y = y.flatten()
    acc = 0
    for i in range(y.size):
        if (predict_y[i] >= 0.5) and (y[i] == 1):
            acc += 1
        if (predict_y[i] < 0.5) and (y[i] == 0):
            acc += 1
    print('accuracy: ' + str(acc) + '/' + str(y.size))


def uci_test():
    """
    uci数据测试
    :return:
    """
    f = open('breast-cancer.data', 'r')
    data = []
    lines = f.readlines()
    dic = dict()
    cnt = np.zeros(10)
    for line in lines:
        line = line.strip()
        line = line.split(',')
        for i in range(len(line)):
            if dic.get(line[i], 0) == 0:
                cnt[i] += 1
            dic[line[i]] = int(cnt[i])
            line[i] = int(cnt[i])
        data.append(line)
    data = np.array(data)
    np.random.shuffle(data)
    train_x = data[0:150:, :-1:]
    train_y = data[0:150:, -1::]
    test_x = data[150::, :-1:]
    test_y = data[150::, -1::]
    theta = gradient_descent(train_x, train_y)
    print(theta)
    get_accuracy(theta, test_x, test_y)


def plot(theta, start, end):
    x1 = np.linspace(start, end, 1000)
    theta = -theta / theta[2][0]
    func = np.poly1d(np.array(theta.T)[0][-2::-1])
    y1 = func(x1)
    plt.plot(x1, y1)
    plt.show()


def work():
    """
    二元测试
    :return:
    """
    loc1 = np.array([1, 1])
    loc2 = np.array([3, 3])
    scale1 = 0.5 * np.ones(2)
    scale2 = 0.5 * np.ones(2)
    size1 = 100
    size2 = 100
    data_set, y = generate_data(loc1, scale1, size1, loc2, scale2, size2)
    # print(_data_set, _y)
    theta = gradient_descent(data_set, y, 0.01)
    print(theta)
    """
    theta0 + theta1 x + theta2 y = 0
    y = -theta1 / theta2 x - theta0 / theta2
    """
    if loc1.size == 2:
        plot(theta, 0, 4)


def generate_related_data(mean1, cov1, size1, mean2, cov2, size2):
    """
    生成不独立的数据
    :param mean1: 类别1均值
    :param cov1: 类别1的2个特征协方差矩阵
    :param size1: 类别1的数据个数
    :param mean2: 类别2均值
    :param cov2: 类别2的2个特征协方差矩阵
    :param size2: 类别2的数据个数
    :return:
    """
    x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=size1)
    x2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=size2)
    plt.plot(x1[:, 0:1], x1[:, 1:], '.', color='red')
    plt.plot(x2[:, 0:1], x2[:, 1:], '.', color='blue')
    y1 = np.ones(size1)
    y2 = np.zeros(size2)
    x1 = np.column_stack((x1, y1))
    x2 = np.column_stack((x2, y2))
    x1 = np.row_stack((x1, x2))
    np.random.shuffle(x1)
    return x1[:, :-1], x1[:, -1:]


def related_test():
    """
    相关数据测试
    :return:
    """
    mean1 = np.array([1, 1])
    mean2 = np.array([5, 5])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    cov2 = np.array([[1, 0.5], [0.5, 1]])
    x, y = generate_related_data(mean1, cov1, 100, mean2, cov2, 100)
    theta = gradient_descent(x, y)
    plot(theta, -2, 8)


if __name__ == '__main__':
    # uci_test()
    work()
    # related_test()


