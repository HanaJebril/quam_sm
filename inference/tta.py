import torch
import torch.nn.functional as F
from .test_data_augmentations import *


def get_tta_transforms(
        crop_h=224,
        crop_w=224,
        scale_factor=1.2,
        noise_std=0.05,
        blur_kernel=5,
        blur_sigma=1.0,
        sharp_factor=2.0,
        bright_factor=1.5
):
    """
    Returns a list of 10 TTA transforms (augment_fn, deaugment_fn).
    """
    return [
        (identity_augment, identity_deaugment),
        (hflip_augment, hflip_deaugment),
        (vflip_augment, vflip_deaugment),
        (rotate90_augment, rotate90_deaugment),
        (lambda x: gaussian_noise_augment(x, std=noise_std), identity_deaugment),
#         (lambda x: gaussian_blur_augment(x, kernel_size=blur_kernel, sigma=blur_sigma), identity_deaugment),
#         (lambda x: sharpness_augment(x, sharpness_factor=sharp_factor), identity_deaugment),
#         (lambda x: brightness_augment(x, brightness_factor=bright_factor), identity_deaugment)
    ]


def tta_predict_10_augs(model, x, tta_transforms):
    """
    Perform TTA inference with geometric and intensity augmentations.

    Args:
        model (nn.Module): segmentation model
        x (torch.Tensor): input (B, C, H, W)
        tta_transforms (list): list of (augment_fn, deaugment_fn)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Stacked probabilities (N, B, C, H, W)
            - Stacked argmax predictions (N, B, H, W)
    """
    model.eval()
    prob_list = []
    argmax_list = []
    
    with torch.no_grad():
        output_ref = model(x)
        prob_ref = F.softmax(output_ref, dim=1)
        for aug_fn, deaug_fn in tta_transforms:
            x_aug, meta = aug_fn(x)
            out = model(x_aug)
            prob = F.softmax(out, dim=1)
            pred_argmax = torch.argmax(out, dim=1)
            prob_deaug = deaug_fn(prob, meta)
            argmax_deaug = deaug_fn(pred_argmax.unsqueeze(1), meta)
            prob_list.append(prob_deaug)
            argmax_list.append(argmax_deaug.squeeze(1))

    all_probs = torch.cat(prob_list, dim=0)
    all_argmax = torch.cat(argmax_list, dim=0)
    return all_probs, all_argmax, prob_ref
