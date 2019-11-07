import numpy as np
import random
import kmeans


def rand_params(dim, k):
    """
    随机初始参数
    :param dim:
    :param k:
    :return:
    """
    mu = np.random.rand(k, dim).reshape(k, dim)
    sigma = np.array([np.eye(dim)] * k).reshape(k, dim, dim)
    alpha = (np.ones(k) * (1.0 / k)).reshape(k, 1)
    return mu, sigma, alpha


def get_probability(x, mu, sigma, threshold=1e-8):
    """
    计算概率
    :param x:
    :param mu:
    :param sigma:
    :param threshold:
    :return:
    """
    n = mu.shape[1]
    if np.linalg.det(sigma) == 0:
        for i in range(sigma.shape[0]):
            sigma[i, i] += threshold
    p = np.exp(-0.5 * np.dot(np.dot(x - mu, np.linalg.pinv(sigma)), (x - mu).T))
    p = p / (np.power(2 * np.pi, n / 2.0) * np.sqrt(np.linalg.det(sigma)))
    return p


def gmm(x, k):
    """
    GMM
    :param x:
    :param k:
    :return:
    """
    x_size = np.shape(x)[0]
    dim = np.shape(x)[1]
    mu, sigma, alpha = rand_params(dim, k)
    gamma = np.zeros((x_size, k))
    lld = np.array([])
    last_l_theta = 1e9
    t = 0
    for times in range(1000):
        prob = np.zeros((x_size, k))
        for i in range(x_size):
            for j in range(k):
                prob[i, j] = get_probability(x[i, :].reshape(1, -1), mu[j, :].reshape(1, -1), sigma[j])
        # E步
        for i in range(k):
            gamma[:, i] = alpha[i, 0] * prob[:, i]
        # 计算似然值
        l_theta = np.sum(np.log(np.sum(gamma, axis=1)))
        if np.abs(last_l_theta - l_theta) < 1e-10:
            t += 1
        else:
            t = 0
        if t == 10:
            print('已收敛，迭代次数{}'.format(times + 1))
            break
        last_l_theta = l_theta
        print(l_theta)
        lld = np.append(lld, l_theta)
        for i in range(x_size):
            gamma[i, :] /= np.sum(gamma[i, :])
        # M步
        alpha = (np.sum(gamma, axis=0) / x_size).reshape(k, 1)
        for i in range(k):
            nk = np.sum(gamma[:, i])
            mu[i, :] = np.dot(gamma[:, i].reshape(1, x_size), x) / nk
            tmp = np.zeros((dim, dim))
            for j in range(x_size):
                v = (x[j, :] - mu[i, :]).reshape(-1, 1)
                tmp += (gamma[j, i] * np.dot(v, v.T))
            sigma[i, :, :] = tmp / nk
    return gamma


def get_y(gamma):
    return np.argmax(gamma, axis=1)


def uci_test():
    with open('bezdekIris.data', 'r') as f:
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            line = line.strip().split(',')
            for word in line[:-1]:
                x.append(float(word))
            if line[-1] == 'Iris-setosa':
                y.append(0)
            elif line[-1] == 'Iris-versicolor':
                y.append(1)
            else:
                y.append(2)
        x = np.array(x).reshape(len(y), -1)
        y = np.array(y).reshape(-1, 1)
        print(x.shape)
        print(y.shape)
        gmm_y = get_y(gmm(x, 3))
        print('real data:')
        for i in range(3):
            print('num of {}: {}'.format(i, np.sum(y == i)))
        print('gmm result:')
        for i in range(3):
            print('num of {}: {}'.format(i, np.sum(gmm_y == i)))


if __name__ == '__main__':
    # x = kmeans.generate_data()
    # gamma = gmm(x, 4)
    # y = get_y(gamma)
    # kmeans.plot(x, y)
    uci_test()
