import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF



def hflip_augment(x):
    # x shape: (B, C, H, W)
    x_flip = torch.flip(x, dims=[3])  # flip along width
    # no separate meta needed; just return the flipped
    return x_flip, None

def hflip_deaugment(pred, meta):
    # pred could be (B, C, H, W) for probabilities or (B, H, W) for argmax
    return torch.flip(pred, dims=[3])


def vflip_augment(x):
    x_flip = torch.flip(x, dims=[2])  # flip along height
    return x_flip, None

def vflip_deaugment(pred, meta):
    return torch.flip(pred, dims=[2])


def rotate90_augment(x):
    # Rotate 90 deg clockwise
    x_rot = x.rot90(k=1, dims=[2, 3])
    return x_rot, None

def rotate90_deaugment(pred, meta):
    # Rotate 90 deg counterclockwise
    pred_rot = pred.rot90(k=-1, dims=[2, 3])
    return pred_rot



def zoom_in_augment(x, scale_factor=1.2):
    """
    "Zoom in" by scaling up and then center-cropping back to the original size.

    Args:
        x (torch.Tensor): shape (B, C, H, W)
        scale_factor (float): e.g., 1.2 for +20% zoom

    Returns:
        x_zoom (torch.Tensor): shape (B, C, H, W), same as input but "zoomed"
        meta: for TTA pipelines, returns None (since shape is unchanged)
    """
    B, C, H, W = x.shape

    # Scale up to a larger intermediate size
    newH = int(H * scale_factor)
    newW = int(W * scale_factor)
    x_scaled = F.interpolate(x, size=(newH, newW), mode='bilinear', align_corners=False)

    # Center-crop back to original H, W
    top = (newH - H) // 2
    left = (newW - W) // 2
    x_zoom = x_scaled[:, :, top: top + H, left: left + W]

    return x_zoom, None

def zoom_in_deaugment(pred, meta):
    """
    Inverse of zoom_in_augment.
    Args:
        pred: (B, C, newH, newW) or (B, newH, newW)
        meta: (H, W) original size
    """
    return pred


def center_crop_augment(x, crop_h=224, crop_w=224):
    """
    x: (B, C, H, W)
    Returns cropped x, plus metadata for de-crop.
    """
    B, C, H, W = x.shape
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2

    x_cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    meta = (top, left, H, W)
    return x_cropped, meta


def center_crop_deaugment(pred, meta):
    """
    Inverse center-crop.
    pred: (B, C, crop_h, crop_w) or (B, crop_h, crop_w)
    meta: (top, left, H, W)
    """
    top, left, H, W = meta
    if pred.ndim == 4:
        B, C, h, w = pred.shape
        out = torch.zeros((B, C, H, W), dtype=pred.dtype, device=pred.device)
        out[:, :, top:top + h, left:left + w] = pred
    else:
        B, h, w = pred.shape
        out = torch.zeros((B, H, W), dtype=pred.dtype, device=pred.device)
        out[:, top:top + h, left:left + w] = pred
    return out



# 7) Gaussian Noise
def gaussian_noise_augment(x, std=0.05):
    """
    Add i.i.d. Gaussian noise to each pixel:
    x: (B, C, H, W)
    """
    noise = torch.randn_like(x) * std
    x_noisy = x + noise
    return x_noisy, None

def identity_deaugment(pred, meta):
    # For noise, no shape/orientation changed, so do nothing
    return pred

# 8) Gaussian Blur (you can tune kernel_size & sigma)
def gaussian_blur_augment(x, kernel_size=5, sigma=1.0):
    """
    x: (B, C, H, W)
    We'll apply blur per image in the batch.
    """
    x_blur_list = []
    for b in range(x.shape[0]):
        # TF.gaussian_blur expects a single image: shape (C, H, W)
        img = x[b]
        img_blur = TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
        x_blur_list.append(img_blur.unsqueeze(0))
    x_blur = torch.cat(x_blur_list, dim=0)
    return x_blur, None

# 9) Sharpness
def sharpness_augment(x, sharpness_factor=2.0):
    """
    >1.0 for more sharp, <1.0 for less sharp.
    """
    x_sharp_list = []
    for b in range(x.shape[0]):
        img = x[b]
        img_sharp = TF.adjust_sharpness(img, sharpness_factor)
        x_sharp_list.append(img_sharp.unsqueeze(0))
    x_sharp = torch.cat(x_sharp_list, dim=0)
    return x_sharp, None

# 10) Brightness
def brightness_augment(x, brightness_factor=1.5):
    """
    brightness_factor=1 => no change, >1 => brighter, <1 => dimmer
    """
    x_bright_list = []
    for b in range(x.shape[0]):
        img = x[b]
        img_bright = TF.adjust_brightness(img, brightness_factor)
        x_bright_list.append(img_bright.unsqueeze(0))
    x_bright = torch.cat(x_bright_list, dim=0)
    return x_bright, None


def identity_augment(x):
    # no change
    return x, None



def center_crop_resize_augment(x, crop_h=256, crop_w=256, model_h=304, model_w=304):
    """
    1) Center-crop x to (crop_h, crop_w).
    2) Resize the cropped region to (model_h, model_w).

    Args:
        x (torch.Tensor): shape (B, C, H, W).
        crop_h (int): desired crop height.
        crop_w (int): desired crop width.
        model_h (int): height for model input (e.g. 304).
        model_w (int): width for model input (e.g. 304).

    Returns:
        x_resized (torch.Tensor): shape (B, C, model_h, model_w).
        meta (tuple): info needed to invert the transform:
            (top, left, original_H, original_W, crop_h, crop_w, model_h, model_w)
    """
    B, C, H, W = x.shape

    # Compute top-left for center crop
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2

    # 1) Center crop
    x_cropped = x[:, :, top: top + crop_h, left: left + crop_w]

    # 2) Resize to model's required size (304×304)
    x_resized = F.interpolate(x_cropped, size=(model_h, model_w), mode='bilinear', align_corners=False)

    # Store metadata
    meta = (top, left, H, W, crop_h, crop_w, model_h, model_w)
    return x_resized, meta




def center_crop_resize_deaugment(pred, meta):
    """
    Inverse of center_crop_resize_augment.

    Args:
        pred (torch.Tensor):
            - shape (B, C, model_h, model_w) [float dtype => probabilities], or
            - shape (B, 1, model_h, model_w) [long dtype => argmax labels].
        meta (tuple): (top, left, H, W, crop_h, crop_w, model_h, model_w)

    Returns:
        out (torch.Tensor): shape (B, C, H, W) or (B, H, W),
                            aligned with the original input image size.
    """
    (top, left, H, W, crop_h, crop_w, model_h, model_w) = meta

    # Check dtype to decide if we're dealing with probabilities (float) or labels (long)
    if pred.dtype in (torch.float32, torch.float16, torch.float64):
        # ----------------
        # CASE 1: Probability map (float)
        # ----------------
        # 1) Resize back to (crop_h, crop_w) with bilinear
        pred_resized = F.interpolate(
            pred, size=(crop_h, crop_w), mode='bilinear', align_corners=False
        )

        # 2) Paste into original canvas
        B, C, _, _ = pred_resized.shape
        out = torch.zeros((B, C, H, W), dtype=pred_resized.dtype, device=pred_resized.device)
        out[:, :, top: top + crop_h, left: left + crop_w] = pred_resized

    else:
        # ----------------
        # CASE 2: Argmax labels (long)
        # ----------------
        # (B, 1, model_h, model_w) => nearest interpolation
        # If your shape is (B, H, W) without channel dim, it's okay too:
        if pred.ndim == 3:
            # (B, model_h, model_w) => unsqueeze
            pred = pred.unsqueeze(1).float()
        elif pred.ndim == 4:
            # (B, 1, model_h, model_w) => convert to float
            pred = pred.float()
        else:
            raise ValueError(f"Unexpected argmax shape: {pred.shape}")

        # Resize back with nearest
        pred_resized = F.interpolate(
            pred, size=(crop_h, crop_w), mode='nearest'
        )  # => (B, 1, crop_h, crop_w)

        # Convert back to long
        pred_resized = pred_resized.squeeze(1).long()  # => (B, crop_h, crop_w)

        # 2) Paste into original canvas
        B, _, _ = pred_resized.shape
        out = torch.zeros((B, H, W), dtype=pred_resized.dtype, device=pred_resized.device)
        out[:, top: top + crop_h, left: left + crop_w] = pred_resized

    return out
