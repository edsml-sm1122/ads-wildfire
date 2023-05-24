import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from wildfire.utils.train import split, create_pairs, create_dataloader, train  # noqa


@pytest.fixture
def sample_data():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return arr


def test_split(sample_data):
    arr = sample_data[:6]
    chunk_size = 2
    expected_result = np.array([1, 2, 3, 4, 5])
    assert np.concatenate(split(arr, chunk_size), axis=0).all() ==  expected_result.all() # noqa


def test_create_pairs(sample_data):
    arr = sample_data
    chunk_size = 2
    expected_result = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
    assert np.concatenate(create_pairs(arr, chunk_size)).all() == expected_result.all()  # noqa


if __name__ == '__main__':
    pytest.main()
