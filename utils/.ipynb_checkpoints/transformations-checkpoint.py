import torch
from torchvision import transforms

# Forward transformation for preprocessing before inference
forward_transform = transforms.Compose([
    transforms.Resize((320, 320), antialias=True),  # Resize to match model input
])

# Inverse transformation to map predictions back to original shape (if needed)
inverse_transform = transforms.Compose([
    transforms.Resize((304, 304), antialias=True),  # Resize to original dataset resolution
])
