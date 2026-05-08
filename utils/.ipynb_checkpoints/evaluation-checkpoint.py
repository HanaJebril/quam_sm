import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from medpy import metric
import pickle
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from .visualization import plot_and_save_images, plot_segmentation_results_final
from .transformations import forward_transform, inverse_transform
# from inference import tta, ensemble, mc_dropout
from utils.seeding import fix_seeds
import cv2
import matplotlib.pyplot as plt
# Set global seed
fix_seeds(42)



def dice_coefficient_np(pred, target, smooth=1e-6):
    """Compute Dice between two binary masks [H,W]."""
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    intersection = np.sum(pred * target)
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice, pred.sum(), target.sum()


def hausdorff_distance_np(pred, target):
    """Compute HD between two binary masks. Penalize empty-vs-nonempty with max distance."""
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    if pred.sum() > 0 and target.sum() > 0:
        return metric.binary.hd(pred, target)
    elif pred.sum() == 0 and target.sum() == 0:
        return 0.0
    else:
        # one empty, one not → max penalty = image diagonal
        H, W = pred.shape
        return np.sqrt(H**2 + W**2)




def dice_coefficient(y_true, y_pred, epsilon=1e-7):
    """
    Compute the Dice coefficient between two binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1).
        y_pred (np.ndarray): Predicted binary mask (0 or 1).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    sum_masks = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + epsilon) / (sum_masks + epsilon)


def hausdorff_distance(y_true, y_pred):
    """
    Compute the symmetric Hausdorff distance between two binary masks.

    Args:
        y_true (np.ndarray): Ground truth binary mask (0 or 1).
        y_pred (np.ndarray): Predicted binary mask (0 or 1).

    Returns:
        float: Hausdorff distance, or np.inf if either mask is empty.
    """
    coords_true = np.argwhere(y_true)
    coords_pred = np.argwhere(y_pred)

    if coords_true.size == 0 or coords_pred.size == 0:
        return np.inf

    forward_hd = directed_hausdorff(coords_true, coords_pred)[0]
    backward_hd = directed_hausdorff(coords_pred, coords_true)[0]
    return max(forward_hd, backward_hd)


def binary_cross_entropy(target, output, epsilon=1e-7):
    """
    Compute binary cross-entropy between two binary arrays.

    Args:
        target (np.ndarray): Ground truth (0 or 1).
        output (np.ndarray): Prediction (0 to 1).

    Returns:
        float: BCE loss.
    """
    output = np.clip(output, epsilon, 1 - epsilon)
    bce = -(target * np.log(output) + (1 - target) * np.log(1 - output))
    return np.mean(bce)


def hamming_distance(target, output):
    """
    Compute the Hamming distance between two binary arrays.

    Args:
        target (np.ndarray): Ground truth (0 or 1).
        output (np.ndarray): Prediction (0 or 1).

    Returns:
        float: Fraction of differing elements.
    """
    return np.mean(target != output)


def compute_uncertainty(class_probabilities, class_probabilities_max):
    """
    Compute uncertainty (entropy and variance) from Monte Carlo dropout predictions.

    Args:
        class_probabilities (torch.Tensor): Stacked class probabilities from multiple forward passes (num_samples, batch_size, channels, height, width).
        class_probabilities_max (torch.Tensor): Stacked max class probabilities for each forward pass.

    Returns:
        tuple: Tuple containing uncertainty estimates - entropy, variance, mean_entropy, mean_variance, normalized_area, normalized_area_fraction.
    """
    # Compute mean class probabilities across samples
    mean_class_probabilities = torch.mean(class_probabilities, dim=0)
    mean_class_probabilities_max = torch.mean(class_probabilities_max.float(), dim=0)

    # Compute entropy for each pixel across classes
    # Entropy measures uncertainty
    entropy = -torch.sum(mean_class_probabilities * torch.log(mean_class_probabilities + 1e-9),
                         dim=0)  # Adding epsilon to prevent log(0)

    # Compute variance across samples for each pixel
    variance = torch.mean(class_probabilities * (1 - class_probabilities), dim=0).sum(dim=0)

    # Calculate the mean entropy and mean variance over all pixels
    mean_entropy = torch.mean(entropy)
    mean_variance = torch.mean(variance)

    # Step 1: Compute the mean prediction map
    mean_prediction_map = torch.mean(mean_class_probabilities, dim=0)  # Shape: (H, W)

    # Step 2: Normalize the uncertainty map (entropy or variance)
    normalized_entropy = entropy / (mean_prediction_map + 1e-6)
    normalized_variance = variance / (mean_prediction_map + 1e-6)

    # Step 3: Compute the normalized area (sum of all normalized pixels)
    normalized_entropy_area = normalized_entropy.sum()
    normalized_variance_area = normalized_variance.sum()

    # Step 4: Optionally normalize by the total number of pixels
    total_pixels = entropy.numel()  # Total number of pixels
    normalized_entropy_fraction = normalized_entropy_area / total_pixels
    normalized_variance_fraction = normalized_variance_area / total_pixels

    return (class_probabilities, mean_class_probabilities, entropy, variance,
            mean_entropy, mean_variance,
            normalized_entropy_area, normalized_entropy_fraction,
            normalized_variance_area, normalized_variance_fraction)


# def compute_dice_and_hd(pred_samples, groundtruth, debug=False):
#     """
#     Compare stochastic segmentation samples against their mean prediction and ground-truth consensus.

#     Args:
#         pred_samples (np.ndarray or torch.Tensor): [num_samples, C, H, W] class scores.
#         groundtruth (np.ndarray or torch.Tensor): [N_annotators, H, W] binary masks.
#         debug (bool): print shapes if True.

#     Returns:
#         dict: Dice/HD statistics and mean prediction vs consensus Dice.
#     """
#     # --- convert to numpy ---
#     if isinstance(pred_samples, torch.Tensor):
#         pred_samples = pred_samples.cpu().numpy()
#     if isinstance(groundtruth, torch.Tensor):
#         groundtruth = groundtruth.cpu().numpy()

#     num_samples, C, H, W = pred_samples.shape
#     if debug:
#         print(f"pred_samples: {pred_samples.shape}, groundtruth: {groundtruth.shape}")

#     # --- consensus GT (majority vote) ---
#     gt_majority = (groundtruth.mean(axis=0) >= 0.5).astype(np.uint8)  # [H,W]

#     # --- mean prediction ---
#     pred_mean = np.mean(pred_samples, axis=0, keepdims=True)  # [1,C,H,W]
#     pred_mean_argmax = np.argmax(pred_mean, axis=1).squeeze(0)  # [H,W]

#     if debug:
#         print("pred_mean_argmax:", pred_mean_argmax.shape, "gt_majority:", gt_majority.shape)

#     # --- per-sample vs mean ---
#     dice_scores, hd_scores = [], []
#     for s in range(num_samples):
#         sample_argmax = np.argmax(pred_samples[s], axis=0)  # [H,W]
#         dice, _, _ = dice_coefficient_np(sample_argmax, pred_mean_argmax)
#         hd = hausdorff_distance_np(sample_argmax, pred_mean_argmax)
#         dice_scores.append(dice)
#         hd_scores.append(hd)

#     # --- mean vs ground-truth consensus ---
#     gt_dice, pred_sum, gt_sum = dice_coefficient_np(pred_mean_argmax, gt_majority)

#     return {
#         "Dice_Statistics": compute_statistics(dice_scores),   # samples vs mean
#         "Hausdorff_Distance_Statistics": compute_statistics(hd_scores),
#         "GT_Dice": gt_dice,           # mean prediction vs majority-vote GT
#         "pred_sum": pred_sum,         # number of predicted positives
#         "gt_sum": gt_sum,             # number of GT positives
#         "pred_mean_argmax": pred_mean_argmax  # [H,W] mask
#     }







def compute_statistics(values):
    """Return mean, median, std, var, min, max."""
    values = np.array(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

def _majority_vote_multiclass(annotators_hw):
    """
    annotators_hw: [A,H,W] int labels in {0..C-1} (or {0,1,2}).
    Returns [H,W] int majority label (ties broken by smallest label).
    """
    if isinstance(annotators_hw, torch.Tensor):
        # torch.mode is fast and stable
        mv = torch.mode(annotators_hw.long(), dim=0).values  # [H,W]
        return mv.cpu().numpy()
    ann = np.asarray(annotators_hw)
    # Convert to torch for an efficient mode; avoids scipy.stats.mode deprecation quirks
    return torch.mode(torch.as_tensor(ann).long(), dim=0).values.cpu().numpy()

def _dice_binary_np(pred_bin, tgt_bin, smooth=1e-6):
    """Dice between two binary masks [H,W] with smoothing."""
    pred = (pred_bin > 0).astype(np.uint8)
    tgt  = (tgt_bin  > 0).astype(np.uint8)
    inter = np.sum(pred * tgt)
    union = pred.sum() + tgt.sum()
    return (2.0 * inter + smooth) / (union + smooth)

def _hd_binary_np(pred_bin, tgt_bin):
    """Hausdorff distance between two binary masks [H,W]."""
    pred = (pred_bin > 0).astype(np.uint8)
    tgt  = (tgt_bin  > 0).astype(np.uint8)
    if pred.sum() > 0 and tgt.sum() > 0:
        return float(metric.binary.hd(pred, tgt))
    elif pred.sum() == 0 and tgt.sum() == 0:
        return 0.0
    else:
        H, W = pred.shape
        return float(np.sqrt(H**2 + W**2))  # penalize empty vs non-empty

    
    
def compute_dice_and_hd_binary(pred_samples, groundtruth, debug=False):
    """
    Binary version (background=0, disc=1).
    Args:
        pred_samples: [S, 2, H, W] softmax probs or logits
        groundtruth: [A, H, W] or [H, W] with {0,1}
    """
    if isinstance(pred_samples, torch.Tensor):
        pred_samples = pred_samples.detach().cpu().numpy()
    if isinstance(groundtruth, torch.Tensor):
        groundtruth = groundtruth.detach().cpu().numpy()

    num_samples, C, H, W = pred_samples.shape
    assert C == 2, f"Expected 2 classes (binary), got {C}"

    # --- consensus GT (majority vote if multiple annotators) ---
    if groundtruth.ndim == 3:  # [A,H,W]
        gt_mv = (groundtruth.mean(axis=0) >= 0.5).astype(np.uint8)
    else:  # [H,W]
        gt_mv = (groundtruth > 0.5).astype(np.uint8)

    # --- mean prediction ---
    pred_mean = np.mean(pred_samples, axis=0)  # [2,H,W]
    pred_mean_argmax = np.argmax(pred_mean, axis=0).astype(np.uint8)  # [H,W]

    # --- per-sample vs mean ---
    dice_scores, hd_scores = [], []
    for s in range(num_samples):
        sample_argmax = np.argmax(pred_samples[s], axis=0).astype(np.uint8)
        dice = dice_coefficient(gt_mv, sample_argmax)
        hd   = hausdorff_distance_np(sample_argmax, pred_mean_argmax)
        dice_scores.append(dice)
        hd_scores.append(hd)

    # --- mean vs GT ---
    gt_dice = dice_coefficient(gt_mv, pred_mean_argmax)

    return {
        "Dice_Statistics": compute_statistics(dice_scores),
        "Hausdorff_Distance_Statistics": compute_statistics(hd_scores),
        "GT_Dice": gt_dice,
        "pred_mean_argmax": pred_mean_argmax
    }
    
    

def dice_hd_multiclass(pred_labels, tgt_labels, num_classes=None):
    """
    Compute per-class Dice/HD and macro-averaged scores for label maps [H,W].
    Returns:
      dice_per_class: list length C
      hd_per_class: list length C
      dice_macro: float
      hd_macro: float
    """
    pred = np.asarray(pred_labels)
    tgt  = np.asarray(tgt_labels)
    if num_classes is None:
        num_classes = int(max(pred.max(), tgt.max())) + 1

    dice_list, hd_list = [], []
    for c in range(num_classes):
        pred_c = (pred == c)
        tgt_c  = (tgt  == c)
        dice_c = _dice_binary_np(pred_c, tgt_c)
        hd_c   = _hd_binary_np(pred_c, tgt_c)
        dice_list.append(float(dice_c))
        hd_list.append(float(hd_c))

    dice_macro = float(np.mean(dice_list)) if len(dice_list) else 0.0
    hd_macro   = float(np.mean(hd_list)) if len(hd_list) else 0.0
    return dice_list, hd_list, dice_macro, hd_macro

def compute_dice_and_hd(pred_samples, groundtruth, debug=False):
    """
    Unified Dice + HD computation.
    Works for both binary (C=2) and multiclass segmentation.

    Args:
        pred_samples (np.ndarray or torch.Tensor): [S, C, H, W] class scores/logits/probs
        groundtruth (np.ndarray or torch.Tensor): [A, H, W] (annotators) or [H, W] (single mask)
        debug (bool): print shapes if True

    Returns:
        dict with metrics:
          - Dice_Statistics (per-sample vs mean, macro Dice)
          - Hausdorff_Distance_Statistics (per-sample vs mean, macro HD)
          - GT_Dice (macro Dice mean prediction vs GT)
          - GT_Dice_per_class (list, one per class)
          - pred_sum_per_class, gt_sum_per_class (pixel counts per class)
          - pred_mean_argmax ([H,W] mean prediction labels)
    """
    # --- convert to numpy ---
    if isinstance(pred_samples, torch.Tensor):
        pred_samples = pred_samples.detach().cpu().numpy()
    if isinstance(groundtruth, torch.Tensor):
        groundtruth = groundtruth.detach().cpu().numpy()

    assert pred_samples.ndim == 4, f"Expected [S,C,H,W], got {pred_samples.shape}"
    num_samples, C, H, W = pred_samples.shape
    if debug:
        print(f"pred_samples: {pred_samples.shape}, groundtruth: {groundtruth.shape}, C={C}")

    # --- consensus GT (majority vote if multiple annotators) ---
    if groundtruth.ndim == 3:  # [A,H,W]
        gt_mv = _majority_vote_multiclass(groundtruth)  # [H,W]
    else:  # [H,W]
        gt_mv = np.asarray(groundtruth).astype(np.int32)

    # --- mean prediction ---
    pred_mean = np.mean(pred_samples, axis=0)  # [C,H,W]
    pred_mean_argmax = np.argmax(pred_mean, axis=0).astype(np.int32)  # [H,W]

    # --- per-sample vs mean ---
    dice_scores, hd_scores = [], []
    for s in range(num_samples):
        sample_argmax = np.argmax(pred_samples[s], axis=0).astype(np.int32)
        dice_list, hd_list, dice_macro, hd_macro = dice_hd_multiclass(
            sample_argmax, pred_mean_argmax, num_classes=C
        )
        dice_scores.append(dice_macro)
        hd_scores.append(hd_macro)

    # --- mean vs GT ---
    dice_per_class, hd_per_class, dice_macro_gt, _ = dice_hd_multiclass(
        pred_mean_argmax, gt_mv, num_classes=C
    )

    # --- summary ---
    return {
        "Dice_Statistics": compute_statistics(dice_scores),
        "Hausdorff_Distance_Statistics": compute_statistics(hd_scores),
        "GT_Dice": dice_macro_gt,
        "GT_Dice_per_class": dice_per_class,
        "pred_sum_per_class": [int((pred_mean_argmax == c).sum()) for c in range(C)],
        "gt_sum_per_class": [int((gt_mv == c).sum()) for c in range(C)],
        "pred_mean_argmax": pred_mean_argmax
    }








# Now you can compute Dice between pred_array and ellipse_mask
def compute_dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    dice = 2. * intersection / (mask1.sum() + mask2.sum())
    
    return dice 
    

def compute_hd(mask1, mask2):
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    if len(coords1) == 0 or len(coords2) == 0:
        return 0.0, 0.0
    hd_1to2 = directed_hausdorff(coords1, coords2)[0]
    hd_2to1 = directed_hausdorff(coords2, coords1)[0]
    return hd_1to2, hd_2to1


def compute_chamfer_distance(mask1, mask2):
    """
    Compute Chamfer distance between two binary masks using OpenCV distance transform.
    Returns the symmetric Chamfer distance.
    """
    # Make sure masks are uint8
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Compute distance transform
    dist1 = cv2.distanceTransform(1 - mask2, cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform(1 - mask1, cv2.DIST_L2, 3)

    chamfer_1to2 = dist1[mask1.astype(bool)].mean() if mask1.sum() > 0 else 0
    chamfer_2to1 = dist2[mask2.astype(bool)].mean() if mask2.sum() > 0 else 0

    return chamfer_1to2 , chamfer_2to1    

def compute_predictions_and_metrics(model, data_loader, device, output_folder, method):
    metrics_data = []

    with torch.no_grad():
        pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
        for itr, (test_images, test_annotations, name) in pbar:
            test_images = (test_images / 255).to(device=device, dtype=torch.float32).squeeze(1)
            test_annotations = test_annotations.to(device=device, dtype=torch.long).squeeze(1)

            if method == "MC":
                predictions, predictions_max = mc_dropout.monte_carlo_dropout_predict(model, test_images, num_samples=10)
            elif method == "deep_ensemble":
                predictions, predictions_max = ensemble.deep_ensemble_predict(model, test_images)
            elif method == "TTA":
                tta_transforms = tta.get_tta_transforms()
                predictions, predictions_max = tta.tta_predict_10_augs(model, test_images, tta_transforms)
            if method == "QUAM_Best":
                predictions, predictions_max = mc_dropout.monte_carlo_dropout_predict(model, test_images, num_samples=10)

            (class_probabilities, mean_class_probabilities, entropy, variance,
             mean_entropy, mean_variance,
             normalized_entropy_area, normalized_entropy_fraction,
             normalized_variance_area, normalized_variance_fraction) = compute_uncertainty(predictions, predictions_max)

            out = compute_dice_and_hd(class_probabilities.cpu().numpy(), test_annotations[0])
#             pred_array = out['pred_mean_argmax'][0, 0].astype(np.float32)
#             pred_tensor = torch.from_numpy(pred_array).to(device=device)
#             transformed_img = forward_transform(pred_tensor.unsqueeze(0).unsqueeze(0))
            
            
            
#             outputs_dae = torch.sigmoid(model_dae(transformed_img))

#             target_np = transformed_img[0, 0].cpu().numpy()
#             output_np_dae = outputs_dae[0, 0].cpu().numpy()

#             target_np = (target_np > 0.5).astype(np.uint8)
#             output_np_dae = (output_np_dae > 0.5).astype(np.uint8)

#             pred_tensor2 = torch.from_numpy(output_np_dae).to(device=device)
#             output_np_gt_dae = inverse_transform(pred_tensor2.unsqueeze(0).unsqueeze(0)).cpu().numpy()
#             output_np_gt_dae = (output_np_gt_dae[0][0] > 0.5).astype(np.uint8)
#             target_np = (pred_tensor.cpu().numpy() > 0.5).astype(np.uint8)

#             dice_score_val_dae = dice_coefficient(target_np, output_np_gt_dae)
#             hausdorff_dist_dae = hausdorff_distance(target_np, output_np_gt_dae)

            plot_segmentation_results_final(
                test_images, test_annotations, out, entropy, variance, output_np_gt_dae, name, output_folder
            )
            
#             dice_score = compute_dice(target_np, output_np_gt_dae)
# #             print("Dice between FAZ segmentation and fitted ellipse:", dice_score)
            
#             hd1, hd2 = compute_hd(target_np, output_np_gt_dae)
#             chamfer1, chamfer2 = compute_chamfer_distance(target_np, output_np_gt_dae)
            
            
            

            

            metrics_row = {
                'Image_Name': name[0],
                'mean_entropy': mean_entropy.item(),
                'mean_variance': mean_variance.item(),
                'normalized_entropy_area': normalized_entropy_area.item(),
                'normalized_entropy_fraction': normalized_entropy_fraction.item(),
                'normalized_variance_area': normalized_variance_area.item(),
                'normalized_variance_fraction': normalized_variance_fraction.item(),
                'dice_mean': out["Dice_Statistics"]["mean"],
#                 'dice_max': out["Dice_Statistics"]["max"],
#                 'dice_min': out["Dice_Statistics"]["min"],
#                 'dice_std': out["Dice_Statistics"]["std"],
#                 'dice_variance': out["Dice_Statistics"]["variance"],
                'hd_mean': out["Hausdorff_Distance_Statistics"]["mean"],
#                 'hd_max': out["Hausdorff_Distance_Statistics"]["max"],
#                 'hd_min': out["Hausdorff_Distance_Statistics"]["min"],
#                 'hd_std': out["Hausdorff_Distance_Statistics"]["std"],
#                 'hd_variance': out["Hausdorff_Distance_Statistics"]["variance"],
                'gt_dice': out['GT_Dice'],
#                 'mean_pred_sum': out['pred_sum_f'],
#                 "Dice_DAE": dice_score_val_dae,
#                 "Hausdorff_DAE": hausdorff_dist_dae,
#                 'Dice_D': dice_score,
#                 'HD1_D': hd1,
#                 'HD2_D': hd2,
#                 'CHD1_D':chamfer1,
#                 'CHD2_D':chamfer2,
#                 "Dice_DAE_ADV": dice_score_val_adv,
#                 "Hausdorff_DAE_ADV": hausdorff_dist_adv
            }

            metrics_data.append(metrics_row)

    return pd.DataFrame(metrics_data)

