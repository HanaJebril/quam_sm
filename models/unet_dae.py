import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple
import numpy as np

class UNetLikeDAE(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        strides: Sequence[int] = (2,),
        image_size: Sequence[int] = (320, 320),
        features: Sequence[int] = (16, 32, 32, 32, 32),
        decoder_features: Sequence[int] = (16, 16, 16, 16, 16),
        intermediate: Sequence[int] = (512, 1024),
        **kwargs
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.decoder_features = decoder_features
        self.intermediate = intermediate
        self.strides = strides

        assert len(features) == len(decoder_features), "For U-Net style skipping, encoder and decoder depths must match."
        self.image_size = image_size
        self.strides = [strides[0]] * len(features) if len(strides) == 1 else strides

        # --- build encoder as a ModuleList, remembering channel sizes for skips ---
        self.encoders = nn.ModuleList()
        enc_ch = in_channels
        for i, (fmap, stride) in enumerate(zip(features, self.strides)):
            is_last = (i == len(features) - 1)
            layer = self._get_encode_layer(enc_ch, fmap, stride, is_last)
            self.encoders.append(layer)
            enc_ch = fmap
        self.skip_channels = list(features)  # will use these in decoder

        # --- compute shape and build bottleneck ---
        rf = int(np.prod(self.strides))
        h, w = image_size[0] // rf, image_size[1] // rf
        self.shape_before_flattening = (features[-1], h, w)
        flattened = features[-1] * h * w
        self.intermediate = self._get_intermediate_module(flattened, intermediate)

         # Build decoder as ModuleList of small ModuleDicts:
        self.decoders = nn.ModuleList()
        reversed_skips = list(reversed(self.features[:-1]))  # skip channels for blocks 0..n-2

        in_ch = self.features[-1]
        for i, (out_ch, stride) in enumerate(zip(self.decoder_features, reversed(self.strides))):
            is_last = (i == len(self.decoder_features) - 1)
            block = nn.ModuleDict({
                # only upsamples the bottleneck (no skip concat yet)
                "upconv": nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride,
                                             padding=1, output_padding=1),
                "act0":   nn.ReLU(),
                # this conv will run *after* we concat the skip
                "conv":   nn.Conv2d(
                             out_ch + (0 if is_last else reversed_skips[i]),
                             (self.out_channels if is_last else out_ch),
                             kernel_size=3, stride=1, padding=1
                         ),
                "act1":   (nn.Identity() if is_last else nn.ReLU())
            })
            self.decoders.append(block)
            in_ch = out_ch

    def forward(self, x):
        # 1) encode and collect skips
        skips = []
        out = x
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)

        # 2) bottleneck
        out = out.view(out.size(0), -1)
        out = self.intermediate(out)
        out = out.view(out.size(0), *self.shape_before_flattening)

        # 3) decode with skip connections
        for i, dec in enumerate(self.decoders):
            out = dec["upconv"](out)
            out = dec["act0"](out)
            # insert skip *before* the conv
            if i < len(self.decoders) - 1:
                skip = skips[-(i + 2)]
                out = torch.cat([out, skip], dim=1)
            out = dec["conv"](out)
            out = dec["act1"](out)

        return out

    
    def _get_encode_module(
            self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        """
        Returns the encode part of the network by building up a sequence of layers returned by `_get_encode_layer`.
        """
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, i == (len(self.features) - 1))
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: Sequence[int]) -> nn.Sequential:
        """
        Returns the intermediate block of the network which accepts input from the encoder and whose output goes
        to the decoder.
        """
        if len(num_inter_units) < 2:
            raise NotImplementedError('There should be at least 2 intermediate layers!')

        intermediate = nn.Sequential()
        layer_channels = in_channels

        for i, units in enumerate(num_inter_units):
            if i != len(num_inter_units) - 1:
                layer = nn.Linear(layer_channels, units)
            else:
                layer = nn.Linear(layer_channels, np.prod(self.shape_before_flattening))
            intermediate.add_module("inter_%i" % i, layer)
            intermediate.add_module("act", nn.ReLU())
            layer_channels = units

        return intermediate

    def _get_decode_module(
            self, in_channels: int, channels: Sequence[int], output_channels: int, strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        """
        Returns the decode part of the network by building up a sequence of layers returned by `_get_decode_layer`.
        """
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, output_channels, i == (len(channels) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, stride: int, is_last: bool) -> nn.Module:
        """
        Returns a single layer of the encoder part of the network.
        """
        layer = nn.Sequential()
        layer.add_module('conv0', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layer.add_module('act0', nn.ReLU())
        if not is_last:
            layer.add_module('conv1', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layer.add_module('act1', nn.ReLU())
        return layer

    def _get_decode_layer(self, in_channels: int, out_channels: int, stride: int, final_out_ch: int,
                          is_last: bool) -> nn.Sequential:
        """
        Returns a single layer of the decoder part of the network.
        """
        layer = nn.Sequential()
        layer.add_module('upconv0', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                                       stride=stride, padding=1, output_padding=1))
        layer.add_module('act0', nn.ReLU())
        if not is_last:
            layer.add_module('conv0', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            layer.add_module('conv0', nn.Conv2d(in_channels, final_out_ch, kernel_size=3, stride=1, padding=1))
#         layer.add_module('act1', nn.Sigmoid())

        return layer

        # A helper to extract encoder features only.
    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns features from the encoder (before flattening) that can be used for uncertainty estimation.
        """
        return self.encode(x)
    
    
    
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=16):
        """
        A simple Conv-based discriminator that outputs a single probability:
          - "1" for real
          - "0" for fake (autoencoder-generated)
        """
        super().__init__()
        self.main = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(num_features, num_features*2, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(num_features*2, num_features*4, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            nn.Conv2d(num_features*4, 1, kernel_size=4, stride=1, padding=0),  # 64 -> 61
            # Global average pooling or flatten + linear could also be used
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_channels, H, W).
        Returns:
            (B, 1) raw logits (you can apply sigmoid later).
        """
        out = self.main(x)             # shape (B, 1, ...)
        # out = torch.flatten(out, 1)    # flatten all except batch dim
        return out

    
    

class ResNetDiscriminator(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')

        # Modify first conv layer to accept 1-channel input
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the classification head
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Keep features only

        # Add your own head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Output shape: (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(512, 1),         # For real/fake decision
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out  # Raw logits (use with BCEWithLogitsLoss)
