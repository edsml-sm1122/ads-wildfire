import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from livelossplot import PlotLosses


def split(arr, chunk_size):
    """
    Split an array into multiple subarrays of equal size.

    Args:
        arr (numpy.ndarray): The input array.
        chunk_size (int): The size of each subarray.

    Returns:
        list: List of subarrays.
    """
    L = len(arr)
    num_splits = L // chunk_size
    remainder = L % chunk_size
    splits = np.split(arr[:L-remainder], num_splits)
    if remainder != 0:
        splits.append(arr[L-remainder:])
    return splits


def create_pairs(data, chunk_size):
    """
    Create input-output pairs from split data.

    Args:
        data (numpy.ndarray): The input data.
        chunk_size (int): The size of each input-output pair.

    Returns:
        tuple: A tuple containing two arrays, input (x) and output (y).

    """
    x = split(data, chunk_size)
    y = split(data, chunk_size)
    for i in range(len(x)):
        x[i] = x[i][:-1]
        y[i] = y[i][1:]
    x, y = np.array(x), np.array(y)
    return np.concatenate(x), np.concatenate(y)


def create_dataloader(path, batch_size, mode='train'):
    """
    Create a DataLoader from split data.

    Args:
        path (str): The path to the data file.
        mode (str, optional): The mode of the dataloader. Defaults to 'train'.

    Returns:
        torch.utils.data.DataLoader: The created DataLoader.

    """
    data = np.array(np.load(open(path, 'rb')))
    data_x, data_y = create_pairs(data, 100)
    tensor_x, tensor_y = torch.Tensor(data_x), torch.Tensor(data_y)
    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode != 'val'))  # noqa
    return data_loader


def train(model, train_data, val_data, epochs=10, patience=3, device='cpu'):
    """
    Trains a model

    Args:
        model (torch.nn.Module): The model to train.
        train_data (torch.utils.data.DataLoader): The DataLoader for training.
        val_data (torch.utils.data.DataLoader): The DataLoader for validation.
        epochs (int, optional): The number of epochs to train. Defaults to 10.
        device (str, optional): The device to use for training
                                (defaults to 'cpu').
        patience (int, optional): The number of epochs to wait for improvement
                                  in validation loss before early stopping.
                                  Defaults to 3.

    Returns:
        torch.nn.Module: The trained autoencoder model.

    """
    m_type = 0
    if model.__class__.__name__ == 'ConvLSTM':
        m_type = 1
        model.double()
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    liveloss = PlotLosses()
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} of {epochs}')
        logs = {}
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        print('Train:')
        for batch, label in tqdm(train_data):
            opt.zero_grad()
            if m_type == 1:
                _, x_hat = model(batch)
                x_hat = x_hat.squeeze()
            else:
                batch = batch.reshape(batch.shape[0], 1, batch.shape[1], batch.shape[2]).to(device) # noqa 
                x_hat, _ = model(batch)
            loss = loss_fn(x_hat.squeeze(), label)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_data)
        logs['loss'] = train_loss
        model.eval()
        print('Val:')
        with torch.no_grad():
            for batch, label in tqdm(val_data):
                if m_type == 1:
                    _, x_hat = model(batch)
                    x_hat = x_hat.squeeze()
                else:
                    batch = batch.reshape(batch.shape[0], 1, batch.shape[1], batch.shape[2]).to(device) # noqa 
                    x_hat, _ = model(batch)
                loss = loss_fn(x_hat.squeeze(), label)
                val_loss += loss.item()
        val_loss /= len(val_data)
        logs['val_loss'] = val_loss
        liveloss.update(logs)
        liveloss.send()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Trained stopped early: No improvement in val loss for {patience} epochs.")  # noqa
                break
    return model
