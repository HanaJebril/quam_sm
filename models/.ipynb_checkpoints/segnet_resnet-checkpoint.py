from torch import nn, Tensor
from typing import List, Optional, Tuple
from torchvision.models import vgg16, VGG16_Weights
import torch
from torchvision.models import resnet50, ResNet50_Weights


__all__ = ["SegNet", "BayesSegNet"]


def _make_layer(
        in_channels: int,
        out_channels: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class EncBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int
    ) -> None:
        super().__init__()
        layers = [_make_layer(in_channels, out_channels)]
        for _ in range(num_layers - 1):
            layers += [_make_layer(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.layers(x)
        x, indices = self.max_pool(x)
        return x, indices


class BayesEncBlock(EncBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int
    ) -> None:
        super().__init__(in_channels, out_channels, num_layers)
        self.dropout = nn.Dropout(0.5, inplace=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, indices = super().forward(x)
        x = self.dropout(x)
        return x, indices


class DecBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int
    ) -> None:
        super().__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        layers = []
        for _ in range(num_layers - 1):
            layers += [_make_layer(in_channels, in_channels)]
        layers += [_make_layer(in_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(
            self, x: Tensor,
            indices: Tensor,
            output_size: Optional[List[int]] = None
    ) -> Tensor:
        x = self.max_unpool(x, indices, output_size)
        x = self.layers(x)
        return x


class BayesDecBlock(DecBlock):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int
    ) -> None:
        super().__init__(in_channels, out_channels, num_layers)
        self.dropout = nn.Dropout(0.5, inplace=False)

    def forward(
            self,
            x: Tensor,
            indices: Tensor,
            output_size: Optional[List[int]] = None
    ) -> Tensor:
        torch.use_deterministic_algorithms(False)
        x = super().forward(x, indices, output_size)
        x = self.dropout(x)
        return x



            
            
class SegNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, vgg_encoder: bool = True) -> None:
        super().__init__()
        self.encoder0 = EncBlock(in_channels, 64, 1)
        self.encoder1 = EncBlock(64, 64, 1)
        self.encoder2 = EncBlock(64, 128, 1)
        self.encoder3 = EncBlock(128, 256, 1)
        self.encoder4 = EncBlock(256, 512, 1)

        self.decoder4 = DecBlock(512, 256, 1)
        self.decoder3 = DecBlock(256, 128, 1)
        self.decoder2 = DecBlock(128, 64, 1)
        self.decoder1 = DecBlock(64, 64, 1)
        self.decoder0 = DecBlock(64, 64, 1)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax2d()

        if vgg_encoder:
            self._init_resnet_encoder()

    # def _init_vgg16_encoder(self):
    #     vgg = vgg16(weights=VGG16_Weights.DEFAULT)
    #     params = []
    #     for module in vgg.modules():
    #         if isinstance(module, nn.Conv2d):
    #             params += [module.state_dict()]
    #
    #     idx = 0
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.Conv2d) and name.startswith("encoder"):
    #             module.load_state_dict(params[idx])
    #             idx += 1
    def _init_resnet_encoder(self):
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_layers = [
            resnet.conv1,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ]
        encoder_blocks = [
            self.encoder0, self.encoder1, self.encoder2, self.encoder3, self.encoder4
        ]

        for i, (resnet_layer, encoder_block) in enumerate(zip(resnet_layers, encoder_blocks)):
            if i == 0:  # Handle the first layer (conv1) separately
                # Adapting ResNet's first layer to match input channels
                weight = resnet_layer.weight  # Shape: [64, 3, 7, 7]
                if weight.size(1) != encoder_block.layers[0][0].in_channels:
                    in_channels = encoder_block.layers[0][0].in_channels
                    new_weight = weight.mean(dim=1, keepdim=True)  # Average weights across channels
                    new_weight = new_weight.repeat(1, in_channels, 1, 1)  # Match the input channels
                    encoder_block.layers[0][0].weight.data = new_weight
                else:
                    encoder_block.layers[0][0].load_state_dict(resnet_layer.state_dict())
            elif isinstance(resnet_layer, nn.Sequential):
                for resnet_sub_layer, segnet_layer in zip(resnet_layer.children(), encoder_block.layers.children()):
                    if isinstance(resnet_sub_layer, nn.Conv2d) and isinstance(segnet_layer, nn.Conv2d):
                        segnet_layer.load_state_dict(resnet_sub_layer.state_dict())
            elif isinstance(resnet_layer, nn.Conv2d):
                encoder_block.layers[0][0].load_state_dict(resnet_layer.state_dict())


    def forward(self, x: Tensor) -> Tensor:
        dim0 = x.size()
        x, indices0 = self.encoder0(x)
        dim1 = x.size()
        x, indices1 = self.encoder1(x)
        dim2 = x.size()
        x, indices2 = self.encoder2(x)
        dim3 = x.size()
        x, indices3 = self.encoder3(x)
        dim4 = x.size()
        x, indices4 = self.encoder4(x)

        x = self.decoder4(x, indices4, dim4)
        x = self.decoder3(x, indices3, dim3)
        x = self.decoder2(x, indices2, dim2)
        x = self.decoder1(x, indices1, dim1)
        x = self.decoder0(x, indices0, dim0)
        x = self.conv(x)
        x = self.softmax(x)
        return x


class BayesSegNet(SegNet):
    def __init__(self, in_channels: int, out_channels: int, vgg_encoder: bool = True) -> None:
        super().__init__(in_channels, out_channels, False)

        # Replace encoder2, encoder3, and encoder4 with Bayesian counterparts
        self.encoder2 = BayesEncBlock(64, 128, 1)  # Matches ResNet's layer2
        self.encoder3 = BayesEncBlock(128, 256, 1) # Matches ResNet's layer3
        self.encoder4 = BayesEncBlock(256, 512, 1) # Matches ResNet's layer4

        # Replace decoder counterparts
        self.decoder4 = BayesDecBlock(512, 256, 1)
        self.decoder3 = BayesDecBlock(256, 128, 1)
        self.decoder2 = BayesDecBlock(128, 64, 1)

        if vgg_encoder:
            self._init_resnet_encoder()
