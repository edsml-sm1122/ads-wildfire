import os
import sys
import pytest
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from wildfire.models import cLSTM # noqa


@pytest.fixture
def convlstmcell():
    input_dim = 1
    hidden_dim = 16
    kernel = (3, 3)
    bias = True
    return cLSTM.ConvLSTMCell(input_dim, hidden_dim, kernel, bias)


@pytest.fixture
def convlstm():
    input_dim = 1
    hidden_dim = 16
    kernel_size = (3, 3)
    num_layers = 2
    return cLSTM.ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers)


def test_convlstmcell_forward(convlstmcell):
    batch_size = 10
    channels = 1
    height = 28
    width = 28
    input_tensor = torch.Tensor(torch.randn(batch_size, channels, height, 
                                            width))
    cur_s = tuple(torch.randn(2, batch_size, 16, height, width))
    h, c = convlstmcell.forward(input_tensor, cur_s)
    assert h.shape == (batch_size, 16, height, width)
    assert c.shape == (batch_size, 16, height, width)


def test_convlstmcell_init_hidden(convlstmcell):
    batch_size = 10
    height = 28
    width = 28
    output = convlstmcell.init_hidden(batch_size, (height, width))
    assert len(output) == 2
    assert len(output[0]) == batch_size
    assert len(output[0][0]) == 16
    assert len(output[0][0][0]) == height
    assert len(output[0][0][0][0]) == width


def test_convlstm__init_hidden(convlstm):
    batch_size = 10
    height = 28
    width = 28
    output = convlstm._init_hidden(batch_size, (height, width))
    assert len(output) == 2
    assert len(output[0]) == 2
    assert len(output[0][0]) == batch_size
    assert len(output[0][0][0]) == 16
    assert len(output[0][0][0][0]) == height
    assert len(output[0][0][0][0][0]) == width


def test_convlstm_forward(convlstm):
    batch_size = 10
    sequence = 3
    channel = 1
    height = 28
    width = 28
    input_tensor = torch.Tensor(torch.randn(batch_size, sequence, channel, 
                                            height, width))
    h, res = convlstm.forward(input_tensor, None)
    assert h.shape == (sequence, 16, height, width)
    assert res.shape == (sequence, 1, height, width)


def test_convlstm__extend_for_multilayer(convlstm):
    param = (3, 3)
    num_layers = 2  
    output = convlstm._extend_for_multilayer(param, num_layers)
    assert output == [(3, 3), (3, 3)]


if __name__ == '__main__':
    pytest.main()


