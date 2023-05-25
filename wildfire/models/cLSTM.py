import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell module.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell module.

        Args:
            input_dim (int): Number of channels of input tensor.
            hidden_dim (int): Number of channels of hidden state.
            kernel_size (int, int): Size of the convolutional kernel.
            bias (bool): Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Perform a forward pass of the ConvLSTM cell.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                                         (batch_size, channels, height, width).
            cur_state (tuple): Tuple containing the current hidden state
                               and cell state.

        Returns:
            tuple: Tuple containing the next hidden state and cell state.
        """
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)  # noqa
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden state and cell state.

        Args:
            batch_size (int): Size of the batch.
            image_size (tuple): Size of the input image (height, width).

        Returns:
            tuple: Tuple containing the initialized hidden state 
                   and cell state.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),  # noqa
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))  # noqa


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM module.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        """
        Initialize the ConvLSTM module.

        Args:
            input_dim (int): Number of channels in the input.
            hidden_dim (list or int): Number of hidden channels.
            kernel_size (list or tuple): Size of the kernel in convolutions.
            num_layers (int): Number of LSTM layers stacked on each other.
            batch_first (bool): Whether or not dimension 0 is the b dimension.
            bias (bool): Bias or no bias in convolution.
            return_all_layers (bool): Return the list of computations
                                      for all layers.
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.h2x = nn.Conv2d(16, 1, 1)
        self.bn = nn.BatchNorm2d(16)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]  # noqa

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Perform a forward pass of the ConvLSTM module.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape:
                                         (b, t, c, h, w) or (t, b, c, h, w).
            hidden_state (list or None): List of initial hidden and cell states
                                         for each layer.
        Returns:
            tuple: Tuple containing two lists of length num_layers
                   (or length 1 if return_all_layers is False):
                1. layer_output_list: List of output tensors for each layer.
                2. last_state_list: List of last hidden states and cell states
                                    for each layer.
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        h_las = last_state_list[-1][0]
        h_las = self.bn(h_las)
        res = self.h2x(h_las)
        return h_las, res

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden states and cell states.

        Args:
            batch_size (int): Size of the batch.
            image_size (tuple): Size of the input image (height, width).

        Returns:
            list: List of tuples, each containing the initialized hidden state
                  and cell state for a layer.
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check if the kernel size is consistent.

        Args:
            kernel_size (tuple or list): Kernel size to check.

        Raises:
            ValueError: If the kernel size is not a tuple or a list of tuples.
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and
                 all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extend the parameter for multiple layers.

        Args:
            param (int or list): Parameter to extend.
            num_layers (int): Number of layers.

        Returns:
            list: Extended parameter list.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
