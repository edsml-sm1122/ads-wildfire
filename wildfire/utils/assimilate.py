import numpy as np
from numpy.linalg import inv


def covariance_matrix(X):
    means = np.array([np.mean(X, axis=1)]).transpose()
    dev_matrix = X - means
    res = np.dot(dev_matrix, dev_matrix.transpose())/(X.shape[1]-1)
    return res


def update_prediction(x, K, H, y):
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def run_assimilation(preds_compr, obs_data_compr):
    R = np.cov(obs_data_compr.T)
    H = np.identity(obs_data_compr.shape[1])
    B = np.cov(preds_compr.T)

    K = KalmanGain(B, H, R)
    updated_data_list = []
    for i in range(len(preds_compr)):
        updated_data = update_prediction(preds_compr[i], K, H, obs_data_compr[i])  # noqa
        updated_data_list.append(updated_data)
    return updated_data_list
