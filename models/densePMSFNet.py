import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#############################################
# 1. Multi-Scale Pyramidal Fusion Module (MSPFM)
#############################################

class MSPFM(nn.Module):
    def __init__(self, in_channels, filters):
        """
        in_channels: number of input channels from the incoming feature map.
        filters: number of output channels for each conv branch.
        """
        super(MSPFM, self).__init__()
        # Convolutions applied directly on the input
        self.conv1x1 = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.conv5x5_1 = nn.Conv2d(in_channels, filters, kernel_size=5, padding=2)
        
        # Branch with average pooling then conv
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3x3_2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        
        self.avgpool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv3x3_3 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        
        self.avgpool6 = nn.AvgPool2d(kernel_size=6, stride=6)
        self.conv3x3_4 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        
        # Final 1x1 convolution after fusion. 
        # The concatenated channels come from: input (in_channels) + 3 branches (each filters)
        self.final_conv = nn.Conv2d(in_channels + 3 * filters, filters, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, C, H, W = x.size()
        # First branch: direct convolutions
        conv1x1 = self.relu(self.conv1x1(x))
        conv3x3_1 = self.relu(self.conv3x3_1(x))
        conv5x5_1 = self.relu(self.conv5x5_1(x))
        
        # Second branch: pooling then convolution then upsample back to (H, W)
        conv3x3_2 = self.relu(self.conv3x3_2(self.avgpool2(x)))
        upsample_2 = F.interpolate(conv3x3_2, size=(H, W), mode='bilinear', align_corners=False)
        
        conv3x3_3 = self.relu(self.conv3x3_3(self.avgpool4(x)))
        upsample_4 = F.interpolate(conv3x3_3, size=(H, W), mode='bilinear', align_corners=False)
        
        conv3x3_4 = self.relu(self.conv3x3_4(self.avgpool6(x)))
        upsample_6 = F.interpolate(conv3x3_4, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension (dim=1 for PyTorch)
        concatenated1 = torch.cat([x, conv1x1, conv3x3_1, conv5x5_1], dim=1)
        concatenated2 = torch.cat([x, upsample_2, upsample_4, upsample_6], dim=1)
        # Elementwise addition of the two concatenated tensors
        concat_output = concatenated1 + concatenated2
        
        final_conv = self.relu(self.final_conv(concat_output))
        return final_conv

#############################################
# 2. sSE, cSE and scSE Blocks
#############################################

class sSEBlock(nn.Module):
    def __init__(self, in_channels):
        super(sSEBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        s = self.sigmoid(self.conv(x))
        return x * s

class cSEBlock(nn.Module):
    def __init__(self, channels):
        super(cSEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.size()
        # Global average pooling over H and W
        y = x.view(B, C, -1).mean(dim=2)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(B, C, 1, 1)
        return x * y

class scSEBlock(nn.Module):
    def __init__(self, channels):
        super(scSEBlock, self).__init__()
        self.sse = sSEBlock(channels)
        self.cse = cSEBlock(channels)
        
    def forward(self, x):
        return self.sse(x) + self.cse(x)

#############################################
# 3. Module with scSE Block
#############################################

class ModuleWithSCSE(nn.Module):
    def __init__(self, in_channels, filters):
        """
        in_channels: number of channels of the input feature map.
        filters: number of output channels.
        """
        super(ModuleWithSCSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.scse = scSEBlock(filters)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x

#############################################
# 4. DensePMSFNet Model (Encoder-Decoder)
#############################################

class DensePMSFNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.0, batch_norm=True):
        super(DensePMSFNet, self).__init__()
        self.filter_num = 64
        self.up_samp_size = 2  # used for consistency with TF code

#         # Backbone: DenseNet121 (using torchvision) – note that we use .features
#         backbone = models.densenet121(pretrained=True)
        
        
        
        
        # Load the DenseNet121 backbone
        backbone = models.densenet121(weights= 'DenseNet121_Weights.DEFAULT')
        # Replace the first convolution layer to accept 1 channel
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)

        # Optionally, initialize new_conv weights by averaging the weights of the original conv over the input channel dimension
        with torch.no_grad():
            new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))

        backbone.features.conv0 = new_conv

        self.backbone = backbone.features
        
        # (Optional) Freeze early layers if desired, e.g.:
        # for name, param in list(self.backbone.named_parameters())[:52]:
        #     param.requires_grad = False

        # Define MSPFM modules for the different stages.
        # The channel numbers below are approximate and based on common DenseNet121 dimensions:
        # (s1: ~64 channels, s2: ~256, s3: ~512, s4: ~1024, b1: ~1024)
        self.mspfm_b1 = MSPFM(in_channels=1024, filters=4 * self.filter_num)   # from b1
        self.mspfm_s4 = MSPFM(in_channels=1024, filters=4 * self.filter_num)   # from s4
        self.mspfm_s3 = MSPFM(in_channels=512, filters=4 * self.filter_num)    # from s3
        self.mspfm_s2 = MSPFM(in_channels=256, filters=2 * self.filter_num)    # from s2
        self.mspfm_s1 = MSPFM(in_channels=64, filters=self.filter_num)         # from s1

        # Decoder: define modules with scSE blocks.
        # After each upsampling, features are concatenated so the input channel counts double.
        self.mod_scse_32 = ModuleWithSCSE(in_channels=(4 * self.filter_num) + (4 * self.filter_num), filters=4 * self.filter_num)
        self.mod_scse_64 = ModuleWithSCSE(in_channels=(4 * self.filter_num) + (4 * self.filter_num), filters=4 * self.filter_num)
        self.mod_scse_128 = ModuleWithSCSE(in_channels=(4 * self.filter_num) + (2 * self.filter_num), filters=2 * self.filter_num)
        self.mod_scse_256 = ModuleWithSCSE(in_channels=(2 * self.filter_num) + (self.filter_num), filters=self.filter_num)

        # Final deep fusion: upsample intermediate decoder outputs to the original resolution and concatenate.
        # Compute the number of channels after concatenation:
        final_in_channels = (4 * self.filter_num) + (4 * self.filter_num) + (2 * self.filter_num) + (self.filter_num)
        self.final_conv = nn.Conv2d(final_in_channels, num_classes, kernel_size=1, padding=0)
        self.final_bn = nn.BatchNorm2d(num_classes)
#         self.final_activation = nn.Sigmoid()  
        self.final_activation = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        # --- Encoder: DenseNet121 backbone ---
        x0 = self.backbone.conv0(x)        # typically output: (B, 64, H/2, W/2)
        x0 = self.backbone.norm0(x0)
        x0 = self.backbone.relu0(x0)
        s1 = x0  # s1 feature map

        x1 = self.backbone.pool0(x0)        # (B, 64, H/4, W/4)
        x1 = self.backbone.denseblock1(x1)   # (B, ~256, H/4, W/4)
        s2 = x1

        x2 = self.backbone.transition1(x1)   # downsample
        x2 = self.backbone.denseblock2(x2)   # (B, ~512, H/8, W/8)
        s3 = x2

        x3 = self.backbone.transition2(x2)   # downsample
        x3 = self.backbone.denseblock3(x3)   # (B, ~1024, H/16, W/16)
        s4 = x3

        x4 = self.backbone.transition3(x3)   # further downsample
        x4 = self.backbone.denseblock4(x4)   # (B, ~1024, H/32, W/32)
        b1 = x4

        # --- Decoder ---
        # Process backbone outputs with MSPFM modules
        sapp_bn = self.mspfm_b1(b1)   # output channels: 4*filter_num
        sapp_4 = self.mspfm_s4(s4)
        sapp_3 = self.mspfm_s3(s3)
        sapp_2 = self.mspfm_s2(s2)
        sapp_1 = self.mspfm_s1(s1)

        # Stage 1: upsample b1 branch and fuse with s4 features.
        up_32 = F.interpolate(sapp_bn, scale_factor=2, mode='bilinear', align_corners=False)
        up_32 = torch.cat([up_32, sapp_4], dim=1)
        up_conv_32 = self.mod_scse_32(up_32)

        # Stage 2: upsample and fuse with s3 features.
        up_64 = F.interpolate(up_conv_32, scale_factor=2, mode='bilinear', align_corners=False)
        up_64 = torch.cat([up_64, sapp_3], dim=1)
        up_conv_64 = self.mod_scse_64(up_64)

        # Stage 3: upsample and fuse with s2 features.
        up_128 = F.interpolate(up_conv_64, scale_factor=2, mode='bilinear', align_corners=False)
        up_128 = torch.cat([up_128, sapp_2], dim=1)
        up_conv_128 = self.mod_scse_128(up_128)

        # Stage 4: upsample and fuse with s1 features.
        up_256 = F.interpolate(up_conv_128, scale_factor=2, mode='bilinear', align_corners=False)
        up_256 = torch.cat([up_256, sapp_1], dim=1)
        up_conv_256 = self.mod_scse_256(up_256)

        # Adjust intermediate decoder outputs to full resolution
        up_conv_32_adjusted = F.interpolate(up_conv_32, scale_factor=16, mode='bilinear', align_corners=False)
        up_conv_64_adjusted = F.interpolate(up_conv_64, scale_factor=8, mode='bilinear', align_corners=False)
        up_conv_128_adjusted = F.interpolate(up_conv_128, scale_factor=4, mode='bilinear', align_corners=False)
        up_conv_256_adjusted = F.interpolate(up_conv_256, scale_factor=2, mode='bilinear', align_corners=False)

        # Deep fusion: concatenate all upsampled features along channel dimension
        deepfusion = torch.cat([up_conv_32_adjusted, up_conv_64_adjusted,
                                  up_conv_128_adjusted, up_conv_256_adjusted], dim=1)
        conv_final = self.final_conv(deepfusion)
        conv_final = self.final_bn(conv_final)
#         conv_final = self.final_activation(conv_final)


        return conv_final

# if __name__ == '__main__':
#     # Example: create model and run a dummy forward pass
#     model = DensePMSFNet(input_channels=1, num_classes=2)
#     x = torch.randn(1, 1, 320, 320)
#     y = model(x)
#     print("Output shape:", y.shape)
