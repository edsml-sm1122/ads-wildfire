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


def test_split_odd(sample_data):
    arr = sample_data[:7]
    chunk_size = 2
    arr_split = split(arr, chunk_size)
    print(arr)
    assert len(arr_split) == 4
    assert np.array(arr_split[0]).all() == np.array([1, 2]).all()
    assert np.array(arr_split[1]).all() == np.array([3, 4]).all()
    assert np.array(arr_split[2]).all() == np.array([5, 6]).all()
    assert np.array(arr_split[3]).all() == np.array([5, 6]).all()


def test_split_even(sample_data):
    arr = sample_data[:6]
    chunk_size = 3
    arr_split = split(arr, chunk_size)
    assert len(arr_split) == 2
    assert np.array(arr_split[0]).all() == np.array([1, 2, 3]).all()
    assert np.array(arr_split[1]).all() == np.array([4, 5, 6]).all()


def test_split_empty():
    arr = np.array([])
    chunk_size = 2
    with pytest.raises(ZeroDivisionError):
        split(arr, chunk_size)


def test_split_zero(sample_data):
    arr = sample_data[:7]
    chunk_size = 0
    with pytest.raises(ZeroDivisionError):
        split(arr, chunk_size)


def test_create_pairs_even(sample_data):
    arr = sample_data
    chunk_size = 2
    arr_pairs = create_pairs(arr, chunk_size)
    assert len(arr_pairs) == 2
    assert np.array(arr_pairs[0]).all() == np.array([1, 3, 5, 7, 9]).all()
    assert np.array(arr_pairs[1]).all() == np.array([2, 4, 6, 8, 10]).all()


def test_create_pairs_odd(sample_data):
    arr = sample_data[:7]
    chunk_size = 2
    with pytest.raises(ValueError):
        create_pairs(arr, chunk_size)


def test_create_pairs_empty():
    arr = np.array([])
    chunk_size = 2
    with pytest.raises(ZeroDivisionError):
        create_pairs(arr, chunk_size)


def test_create_pairs_zero(sample_data):
    arr = sample_data
    chunk_size = 0
    with pytest.raises(ZeroDivisionError):
        create_pairs(arr, chunk_size)


def test_create_dataloader_train():
    data_path = '../wildfire/data/Ferguson_fire_test.npy'
    if not os.path.exists(data_path):
        pytest.skip("Test skipped: Required data file cannot be found.")
    batch_size = 32
    dataloader = create_dataloader(data_path, batch_size)
    batch = next(iter(dataloader))
    assert len(batch) == 2
    assert batch[0].shape == (32, 256, 256)
    assert batch[1].shape == (32, 256, 256)


def test_create_dataloader_val():
    data_path = '../wildfire/data/Ferguson_fire_test.npy'
    if not os.path.exists(data_path):
        pytest.skip("Test skipped: Required data file cannot be found.")
    batch_size = 32
    dataloader = create_dataloader(data_path, batch_size, mode='val')
    batch = next(iter(dataloader))
    assert len(batch) == 2
    assert batch[0].shape == (32, 256, 256)
    assert batch[1].shape == (32, 256, 256)
    assert batch[0][1].numpy().all() == batch[1][0].numpy().all()


def test_create_dataloader_err():
    data_path = '../wildfire/data/Ferguson_fire_background.npy'
    if not os.path.exists(data_path):
        pytest.skip("Test skipped: Required data file cannot be found.")
    batch_size = 32
    with pytest.raises(ZeroDivisionError):
        create_dataloader(data_path, batch_size)


if __name__ == '__main__':
    pytest.main()
