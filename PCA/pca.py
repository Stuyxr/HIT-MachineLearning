import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


def generate_data(dim=3, size=100):
    if dim == 2:
        mean = [1, 2]
        cov = [[0.1, 0], [0, 2]]
    if dim == 3:
        mean = [1, 2, 3]
        cov = [[0.1, 0, 0], [0, 3, 0], [0, 0, 3]]
    data = []
    for index in range(size):
        data.append(np.random.multivariate_normal(mean, cov).tolist())
    data = np.array(data)
    data.reshape(size, dim)
    return data


def pca(data, reduced_dim=2):
    mean = np.mean(data, axis=0)
    data = data - mean
    cov = np.dot(data.T, data)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    idx = np.argsort(-eig_vals, axis=0)[:reduced_dim:]
    # print(np.dot(np.dot(eig_vecs.T, cov), eig_vecs))
    eig_vecs = eig_vecs[:, idx]
    return mean, data, eig_vecs


def recover_data(w, centered_data, mean):
    return np.dot(np.dot(centered_data, w), w.T) + mean


def plot(origin_data, pca_data):
    fig = plt.figure()
    dim = origin_data.shape[1]
    if dim == 3:
        ax = Axes3D(fig)
        ax.scatter(origin_data[:, 0], origin_data[:, 1], origin_data[:, 2], c='r', label='Original Data')
        ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c='b', label='PCA Data')
    if dim == 2:
        plt.axis('equal')
        plt.scatter(origin_data[:, 0], origin_data[:, 1], c='r', label='Original Data')
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c='b', label='PCA Data')
    plt.show()


def calc_snr(source, target):
    diff = source - target
    diff = diff ** 2
    mse = np.sqrt(np.mean(diff))
    return 20 * np.log10(255.0 / mse)


def test(path='./Japanese'):
    data = []
    for i in range(1, 24):
        new_path = path + '/' + str(i) + '.tiff'
        img = np.array(Image.open(new_path).resize((50, 50)).convert('L'), 'f').astype(np.float).flatten()
        data.append(img)
        # w = np.array(w, dtype=np.uint32)
    data = np.array(data).reshape(23, -1)
    mean, centered_data, w = pca(data, 10)
    pca_data = recover_data(w, centered_data, mean)
    pca_data[pca_data < 0] = 0
    # print(pca_data)
    pca_data = pca_data.astype(np.uint8)
    for i in range(1, 24):
        new_data = pca_data[i - 1].reshape(50, 50)
        new_image = Image.fromarray(new_data, mode='L')
        new_image.save(path + '/' + str(i) + '_.tiff')
        print('image {}: snr = {}'.format(i, calc_snr(data[i - 1].flatten(), new_data.flatten())))


if __name__ == '__main__':
    # data = generate_data(dim=3)
    # mean, centered_data, w = pca(data, reduced_dim=2)
    # pca_data = np.dot(np.dot(centered_data, w), w.T) + mean
    # plot(data, pca_data)
    test()

