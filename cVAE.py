import torch
import torch.nn as nn

class VAE_Encoder_Conv(nn.Module):
    def __init__(self):
        """
        Class containing the convolutional encoder (image -> latent).

        Layers:
        - layer1: Conv2d layer with 1 input channel, 20 output channels, kernel size 5, and padding 2,
                    followed by GELU activation and MaxPool2d layer with stride 2
        - layer2: Conv2d layer with 20 input channels, 40 output channels, kernel size 5, and padding 2,
                    followed by GELU activation and MaxPool2d layer with stride 2
        - layer3: Conv2d layer with 40 input channels, 60 output channels, kernel size 3, and padding 1,
                    followed by GELU activation and MaxPool2d layer with stride 2
        - layerMu: Conv2d layer with 60 input channels, 120 output channels, kernel size 3, and padding 1,
                    followed by GELU activation and MaxPool2d layer with stride 2
        - layerSigma: Conv2d layer with 60 input channels, 120 output channels, kernel size 3, and padding 1,
                        followed by GELU activation and MaxPool2d layer with stride 2
        """
        super(VAE_Encoder_Conv, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, 5, padding=2),  # Pad so that image dims are preserved
            nn.GELU(),
            nn.MaxPool2d(2, stride=2)  # Halves the spatial dimensions
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

        Args:
        - x: [float] the image

        Returns:
        - mu: [float] mean of the learned latent space representation
        - sigma: [float] standard deviation of the learned latent space representation
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mu =  self.layerMu(x)
        sigma = self.layerSigma(x)
        return mu, sigma
    
class VAE_Decoder_Conv(nn.Module):  
    def __init__(self):
        """
        Class containing the convolutional decoder (latent -> image).

        Layers:
        - layer1: ConvTranspose2d layer with 120 input channels, 60 output channels, kernel size 4, stride 2, and padding 1,
                  followed by GELU activation
        - layer2: ConvTranspose2d layer with 60 input channels, 40 output channels, kernel size 4, stride 2, and padding 1,
                  followed by GELU activation
        - layer3: ConvTranspose2d layer with 40 input channels, 20 output channels, kernel size 4, stride 2, and padding 1,
                  followed by GELU activation
        - layer4: ConvTranspose2d layer with 20 input channels, 10 output channels, kernel size 4, stride 2, and padding 1,
                  followed by GELU activation
        - layer5: ConvTranspose2d layer with 10 input channels, 1 output channel, kernel size 5, stride 1, and padding 2,
                  followed by Tanh activation
        """
        super(VAE_Decoder_Conv, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(120, 60, 4, stride=2, padding=1),  # Upsample by a factor of 2
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
            nn.ConvTranspose2d(10, 1, 5, stride=1, padding=2),  # Preserve spatial dimensions
            nn.Tanh()
        )  # Dims in 256x256 -> out 256x256

    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
        - x: [float] the latent space representation

        Returns:
        - x: [float] the reconstructed image
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
 
class VAE_Conv(nn.Module):
    def __init__(self, device):
        """
        Class combining the convolutional encoder and the convolutional decoder with a VAE latent space.

        Args:
        - device: [str] the device on which to perform the computations

        Attributes:
        - device: [str] the device on which to perform the computations
        - encoder: [VAE_Encoder_Conv] the VAE encoder network
        - decoder: [VAE_Decoder_Conv] the VAE decoder network
        - distribution: [torch.distributions.Normal] the distribution used for sampling from the latent space
        """
        super(VAE_Conv, self).__init__()
        self.device = device
        self.encoder = VAE_Encoder_Conv()
        self.decoder = VAE_Decoder_Conv()
        self.distribution = torch.distributions.Normal(0, 1)  # Sample from N(0,1)

    def sample_latent_space(self, mu, sigma):
        """
        Sample from the latent space.

        Args:
        - mu: [float] mean of the learned latent space representation
        - sigma: [float] standard deviation of the learned latent space representation

        Returns:
        - z: [float] the sampled latent vector
        - kl_div: [float] the KL divergence term for regularization
        """       
        z = mu + sigma * self.distribution.sample(mu.shape).to(self.device)  # Sample the latent distribution
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()  # A term, which is required for regularisation
        return z, kl_div

    def forward(self, x):
        """
        Forward pass of the Convolutional VAE.

        Args:
        - x: [float] a batch of images from the data-loader

        Returns:
        - z: [float] the reconstructed image
        - kl_div: [float] the KL divergence term for regularization
        """
        mu, sigma = self.encoder(x)  # Run the image through the Encoder
        z, kl_div = self.sample_latent_space(mu, sigma)  # Take the output of the encoder and get the latent vector 
        z = self.decoder(z)  # Return the output of the decoder (the predicted image)
        return z, kl_div