import torch
import torch.nn as nn
from typing import Sequence, Tuple

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (16, 32, 64, 128),  # Encoder channels
        decoder_features: Sequence[int] = (128, 64, 32, 16),  # Decoder channels
        image_size: Sequence[int] = (512, 512),
        strides: Sequence[int] = (2,),  # Strides for downsampling
    ):
        """
        Standard Autoencoder (AE) for Image Reconstruction.

        Args:
            in_channels: Number of input channels (Default: 1 for grayscale images).
            out_channels: Number of output channels (Default: 1).
            features: Feature maps for the encoder.
            decoder_features: Feature maps for the decoder.
            image_size: Resolution of the input image (default: 512x512).
            strides: Strides for downsampling (default: 2 for each layer).
        """
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.decoder_features = decoder_features
        self.strides = [strides[0] for _ in features] if len(strides) == 1 else strides

        # Calculate feature map size after encoding
        rf = 2 ** len(features)  # Downsampling factor
        self.latent_shape = (features[-1], image_size[0] // rf, image_size[1] // rf)

        # **ENCODER**
        self.encoder = self._build_encoder()

        # **DECODER**
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Creates the encoder (convolutional layers for downsampling)."""
        layers = []
        in_channels = self.in_channels

        for out_channels in self.features:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels  # Update channels for next layer

        return nn.Sequential(*layers)

    def _build_decoder(self):
        """Creates the decoder (transposed convolution layers for upsampling)."""
        layers = []
        in_channels = self.features[-1]

        for out_channels in self.decoder_features:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels  # Update channels for next layer

        # Final layer: Output same number of channels as input
        layers.append(nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())  # Output in range [0,1]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        """
        encoded = self.encoder(x)  # Encode to latent space
        reconstructed = self.decoder(encoded)  # Decode back
        return reconstructed

    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns encoder features (latent space representation).
        """
        return self.encoder(x)


