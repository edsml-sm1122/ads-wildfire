import numpy as np
from numpy.linalg import inv


def covariance_matrix(X):
    """
    Calculates the covariance matrix of the input data.

    Args:
        X (numpy.ndarray): Input data matrix with shape (n, m).

    Returns:
        numpy.ndarray: Covariance matrix of shape (n, n).
    """
    means = np.array([np.mean(X, axis=1)]).transpose()
    dev_matrix = X - means
    res = np.dot(dev_matrix, dev_matrix.transpose())/(X.shape[1]-1)
    return res


def update_prediction(x, K, H, y):
    """
    Updates the prediction using the Kalman filter equations.

    Args:
        x (numpy.ndarray): Current state estimate with shape (n, 1).
        K (numpy.ndarray): Kalman Gain matrix with shape (n, p).
        H (numpy.ndarray): Measurement matrix with shape (p, n).
        y (numpy.ndarray): Measurement vector with shape (p, 1).

    Returns:
        numpy.ndarray: Updated state estimate with shape (n, 1).
    """
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    """
    Calculates the Kalman gain matrix.

    Args:
        B (numpy.ndarray): Background error covariance matrix with
                           shape (n, n).
        H (numpy.ndarray): Measurement matrix with shape (p, n).
        R (numpy.ndarray): Measurement error covariance matrix
                           with shape (p, p).

    Returns:
        numpy.ndarray: Kalman gain matrix with shape (n, p).
    """
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def run_assimilation(preds_compr, obs_data_compr):
    """
    Runs the assimilation process.

    Args:
        preds_compr (numpy.ndarray): Matrix of predicted values
                                     with shape (m, n).
        obs_data_compr (numpy.ndarray): Matrix of observed values
                                        with shape (m, p).

    Returns:
        list: List of updated state estimates, each with shape (n, 1).
    """
    R = np.cov(obs_data_compr.T)
    H = np.identity(obs_data_compr.shape[1])
    B = np.cov(preds_compr.T)

    K = KalmanGain(B, H, R)
    updated_data_list = []
    for i in range(len(preds_compr)):
        updated_data = update_prediction(preds_compr[i], K, H, obs_data_compr[i])  # noqa
        updated_data_list.append(updated_data)
    return updated_data_list
