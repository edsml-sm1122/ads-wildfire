import os
import sys
import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from wildfire.utils.assimilate import KalmanGain, covariance_matrix, update_prediction  # noqa


@pytest.fixture
def sample_data():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    K = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    H = np.identity(3)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return X, y, K, H, R, B


def test_covariance_matrix(sample_data):
    X = sample_data[0]
    expected_result = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    assert np.allclose(covariance_matrix(X), expected_result)


def test_update_prediction(sample_data):
    x, K, H, y = sample_data[1:5]
    expected_result = np.array([[-4,  -8, -12], [-10, -23, -36.], [-16, -38, -60]])  # noqa
    assert np.allclose(update_prediction(x, K, H, y), expected_result)


def test_KalmanGain(sample_data):
    B, H, R = sample_data[3:6]
    expected_result = np.array([[-6, -2, 3], [-1, 0.5, 0], [5, 1, -2]])
    assert np.allclose(KalmanGain(B, H, R), expected_result)


if __name__ == '__main__':
    pytest.main()
