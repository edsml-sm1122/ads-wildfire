import torch
import torch.nn as nn


class VAE_Encoder_Conv(nn.Module):
    """
    Convolutional encoder module.
    """
    def __init__(self):
        """
        Initialise the convolutional encoder module (latent -> image).
        """
        super(VAE_Encoder_Conv, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, 5, padding=2),  # Pad to preserve dims
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)  # Halves the spatial dims
        )  # Dims in 256x256 -> out 128x128

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 40, 5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
        )  # Dims in 128x128 -> out 64x64

        self.layer3 = nn.Sequential(
            nn.Conv2d(40, 60, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
        )  # Dims in 64x64 -> out 32x32

        self.layerMu = nn.Sequential(
            nn.Conv2d(60, 120, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
        )  # Dims in 32x32 -> out 16x16

        self.layerSigma = nn.Sequential(
            nn.Conv2d(60, 120, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)
        )  # Dims in 32x32 -> out 16x16

    def forward(self, x):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: float
            the input image.

        Returns
        -------
        mu: float
            mean of the learned latent space representation.
        sigma: float
            stdev of the learned latent space representation.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mu = self.layerMu(x)
        sigma = self.layerSigma(x)
        return mu, sigma


class VAE_Decoder_Conv(nn.Module):
    """
    Convolutional decoder module.
    """
    def __init__(self):
        """
        Initialise the convolutional decoder module (latent -> image).
        """
        super(VAE_Decoder_Conv, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(120, 60, 4, stride=2, padding=1),  # Upsample x2
            nn.GELU()
        )  # Dims in 16x16 -> out 32x32

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(60, 40, 4, stride=2, padding=1),
            nn.GELU()
        )  # Dims in 32x32 -> out 64x64

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(40, 20, 4, stride=2, padding=1),
            nn.GELU()
        )  # Dims in 64x64 -> out 128x128

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(20, 10, 4, stride=2, padding=1),
            nn.GELU()
        )  # Dims in 128x128 -> out 256x256

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(10, 1, 5, stride=1, padding=2),
            nn.Tanh()
        )  # Dims in 256x256 -> out 256x256

    def forward(self, x):
        """
        Forward pass of the decoder.

        Parameters
        ----------
        x: float
            the latent space representation.

        Returns
        -------
        x: float
            the reconstructed image.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class VAE_Conv(nn.Module):
    """
    Convolutional VAE module.
    """
    def __init__(self, device):
        """
        Initilise the convolutional VAE module combining the
        convolutional encoder and the convolutional decoder
        with a VAE latent space.

        Parameters
        ----------
        device: str
            the device on which to perform the computations.
        """
        super(VAE_Conv, self).__init__()
        self.device = device
        self.encoder = VAE_Encoder_Conv()
        self.decoder = VAE_Decoder_Conv()
        self.distribution = torch.distributions.Normal(0, 1)

    def sample_latent_space(self, mu, sigma):
        """
        Sample from the latent space.

        Parameters
        ----------
        mu: float
            mean of the learned latent space.
        sigma: float
            standard deviation of the learned latent space.

        Returns
        -------
        z: float
            the sampled latent vector.
        kl_div: float
            the KL divergence term for regularization.
        """
        epsilon = 1e-8
        sigma = torch.clamp(sigma, epsilon)  # Ensure sigma > 0
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl_div

    def forward(self, x):
        """
        Forward pass of the Convolutional VAE.

        Parameters
        ----------
        x: float
            a batch of images from the data-loader.

        Returns
        -------
        z: float
            the reconstructed image.
        kl_div: float
            the KL divergence term for regularization.
        """
        mu, sigma = self.encoder(x)
        z, kl_div = self.sample_latent_space(mu, sigma)
        z = self.decoder(z)
        return z, kl_div
