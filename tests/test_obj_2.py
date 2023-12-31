import os
import sys
import pytest
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from wildfire.models import cVAE  # noqa


@pytest.fixture
def vae_encoder_conv():
    return cVAE.VAE_Encoder_Conv()


@pytest.fixture
def vae_decoder_conv():
    return cVAE.VAE_Decoder_Conv()


@pytest.fixture
def vae_conv():
    device = 'cpu'
    return cVAE.VAE_Conv(device)


def test_vae_encoder_conv_forward(vae_encoder_conv):
    batch_size = 10
    channels = 1
    height = 28
    width = 28
    x = torch.randn(batch_size, channels, height, width)
    mu, sigma = vae_encoder_conv(x)
    assert mu.shape == (batch_size, 120, 1, 1)
    assert sigma.shape == (batch_size, 120, 1, 1)


def test_vae_decoder_conv_forward(vae_decoder_conv):
    batch_size = 10
    latent_channels = 120
    latent_height = 16
    latent_width = 16
    x = torch.randn(batch_size, latent_channels, latent_height, latent_width)
    output = vae_decoder_conv(x)
    assert output.shape == (batch_size, 1, 256, 256)


def test_vae_conv_sample_latent_space(vae_conv):
    batch_size = 10
    latent_channels = 120
    latent_height = 16
    latent_width = 16
    mu = torch.randn(batch_size, latent_channels, latent_height, latent_width)
    sigma = torch.randn(batch_size, latent_channels, latent_height, latent_width)  # noqa
    z, kl_div = vae_conv.sample_latent_space(mu, sigma)
    assert z.shape == (batch_size, latent_channels, latent_height, latent_width)  # noqa
    assert kl_div.shape == ()


def test_vae_conv_forward(vae_conv):
    batch_size = 10
    channels = 1
    height = 28
    width = 28
    x = torch.randn(batch_size, channels, height, width)
    z, kl_div = vae_conv(x)
    assert z.shape == (batch_size, 1, 16, 16)
    assert kl_div.shape == ()


if __name__ == '__main__':
    pytest.main()
