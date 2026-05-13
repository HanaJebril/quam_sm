import torch
import numpy as np
from typing import Tuple
from torch.nn.functional import one_hot


# --- MIS WEIGHTING FUNCTION (Replaces combine_sample_best) ---
@torch.no_grad()
def combine_sample_mis_weighted(
    average_net_pred: torch.Tensor,  # [B, n_output_features]
    sample_preds: torch.Tensor,      # [B, S, M, C, H, W] (S=samples, M=models, C=classes, HxW=spatial)
    train_loss: torch.Tensor,        # [B, S, M, ...] or [B, S, M, 2] - The penalty loss is used for weighting!
    temperature: float = 1.,
    num_samples: int = None,
    **kwargs
):
    """
    Calculates Mixture Importance Sampling (MIS) weights based on the negative
    exponential of the training loss (Approximating p(w | D)).

    Args:
        average_net_pred: Prediction of the reference network (e.g., BMA).
        sample_preds: Predictions from the adversarial model trajectory.
        train_loss: The loss used for weighting (L_ref(w)). Assumes penalty loss 
                    is used if structure is [B, S, M, 2] (index 1 is selected).
        temperature: Softmax temperature parameter for MIS.
        num_samples: Number of samples (S) to use for weighting.

    Returns:
        Dict containing average_net_pred, sample_preds, and sample_weights (MIS weights).
    """
    eps = 1e-8
    
    if num_samples is None:
        num_samples = train_loss.shape[1]  # use S

    sample_preds = sample_preds[:, :num_samples, :, ...]
    train_loss   = train_loss[:,  :num_samples, ...]

    # 1. Select/Reduce Loss Dimension to [B, S, M]
    if train_loss.ndim > 3:
        # Check if loss contains (objective, penalty) tuple in the last dim (like the user's NumPy code)
        if train_loss.shape[-1] > 1:
            # Select index 1, which corresponds to the penalty loss (L_ref(w))
            train_loss = train_loss[..., 1]
        else:
            # If it's a 4D tensor but the last dim is C or HxW, average across those dims
            train_loss = train_loss.mean(dim=tuple(range(3, train_loss.ndim)))
    elif train_loss.ndim == 4:
        # Fallback if dimensions are [B, S, M, C] and C > 1. This averages across C.
        train_loss = train_loss.mean(dim=-1)

    # Ensure final loss shape is [B, S, M]
    B, S, M = train_loss.shape
    
    # 2. Convert losses to weights via softmax over the S*M trajectory
    #    W_i = softmax(-L_ref(w_i) / T)
    
    # Reshape to [B, S*M] and apply softmax across the whole trajectory for each batch item (B)
    sample_weights = torch.softmax(-train_loss.view(B, -1) / temperature, dim=-1).view(B, S, M)

    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds,
        'sample_weights': sample_weights  # The MIS Weights
    }


@torch.no_grad()
def combine_sample_best(
    average_net_pred: torch.Tensor,  # [n_points, n_output_features] or [B, C, H, W]
    sample_preds: torch.Tensor,      # [n_points, n_samples, n_models, n_output_features] or [B, S, M, C, H, W]
    train_loss,                      # [n_points, n_samples, n_models, ...] (Penalty/Reference Loss)
    model_train_loss,                # [n_samples//n] (Reference model loss used for constraint)
    obj_loss,                        # [n_points, n_samples, n_models, ...] (Objective/Adv Loss)
    window_size: int = 1,
    gamma_slack: float = 1e-2,
    **kwargs
):
    """
    Selects the single best sample (the one minimizing obj_loss) 
    that satisfies the training loss constraint (L_ref(w) < L_ref(w_ref) + gamma).
    
    NOTE: This function assumes that sample_preds is either [B, S, M, C] or [B, S, M, C, H, W].
          It expects loss traces (train_loss, model_train_loss, obj_loss) to be [B, S, M].
    """
    n_points = sample_preds.shape[0]
    n_samples = sample_preds.shape[1]
    n_models = sample_preds.shape[2]
    
    # Handle edge case where n_samples is too small for the window size or is zero.
    # The minimum required n_samples is window_size to form at least one window.
    if n_samples < window_size:
        if n_samples == 0:
            print("Warning: n_samples is 0. Returning the unselected reference prediction.")
        else:
            print(f"Warning: n_samples ({n_samples}) is less than window_size ({window_size}). Selecting only the last available sample.")

        # If n_samples or n_models is 0, we can't select from sample_preds. 
        # Fallback to average_net_pred wrapped in the expected [B, 1, M, ...] output shape.
        if n_samples == 0 or n_models == 0:
            # We must return the average_net_pred wrapped as [B, 1, 1, ...]
            # We use 1 for M because we treat the reference model as one selected prediction.
            if average_net_pred.ndim == 2: # Classification [B, C]
                 sample_preds_out = average_net_pred.unsqueeze(1).unsqueeze(2) # [B, 1, 1, C]
            else: # Segmentation [B, C, H, W]
                 sample_preds_out = average_net_pred.unsqueeze(1).unsqueeze(2) # [B, 1, 1, C, H, W]
            
            return {
                'average_net_pred': average_net_pred,
                'sample_preds': sample_preds_out,
            }
        
        # If n_samples > 0 but < window_size, just select the last sample (S=-1)
        # This will return [B, 1, M, ...]
        return {
            'average_net_pred': average_net_pred,
            'sample_preds': sample_preds[:, -1, :, ...].unsqueeze(1),
        }

    # --- Proceed with complex logic only if n_samples >= window_size ---
    
    # The output prediction tensor dimensions (C, H, W or C)
    pred_dims = sample_preds.shape[3:]

    # --- FIX: Loss Reduction for [B, S, M, 2] inputs ---
    # The comparison logic expects 3D loss tensors [B, S, M]. If the loss has a trailing
    # dimension (e.g., size 2 for [obj_loss, ref_loss]), we must reduce it here.
    if train_loss.ndim > 3 and train_loss.shape[-1] > 1:
        # Assuming index 1 is the reference/penalty loss L_ref(w) for the constraint
        train_loss = train_loss[..., 1]
    
    if obj_loss.ndim > 3 and obj_loss.shape[-1] > 1:
        # Assuming index 0 is the objective/adversarial loss L_adv(w) for minimization
        obj_loss = obj_loss[..., 0]
    # --- END FIX ---
    
    # 1. Prepare loss tensors for windowed comparison
    
    # Ensure model_train_loss is 1D (if it comes as [N, 1] or [N])
    if model_train_loss.ndim == 2 and model_train_loss.shape[1] == 1:
        model_train_loss = model_train_loss.squeeze(-1) # -> [N]
    
    # The reference model loss trace (model_train_loss) must match the length of the adversarial samples (n_samples).
    n_ref_losses = model_train_loss.shape[0]
    
    if n_ref_losses >= n_samples:
        # Case 1: Reference loss is longer or equal to samples (e.g., 100 vs 6). Crop it.
        model_train_loss_repeated = model_train_loss[:n_samples]
    else:
        # Case 2: Reference loss is shorter than samples (e.g., 5 vs 6). Repeat and crop.
        # Calculate ceiling division to ensure we have enough repeats
        n_repeats = (n_samples + n_ref_losses - 1) // n_ref_losses
        model_train_loss_repeated = model_train_loss.repeat(n_repeats)[:n_samples]
        
    # Reshape: [n_samples] -> [1, n_samples] (for batch broadcasting)
    model_train_loss_checkem = model_train_loss_repeated.unsqueeze(0) 

    # Unfold along the sample dimension (dim=1)
    # [1, n_samples] -> [1, n_windows, window_size]
    model_train_loss_checkem = model_train_loss_checkem.unfold(1, window_size, 1)

    # Insert model dimension (dim=2): [1, n_windows, 1, window_size]
    model_train_loss_checkem = model_train_loss_checkem.unsqueeze(2)


    # train_loss_checkem: adversarial loss unfolded
    # [n_points, n_samples, n_models] -> [n_points, n_windows, n_models, window_size]
    train_loss_checkem = train_loss.unfold(1, window_size, 1)
    
    # Reshape reference loss to match train_loss structure for comparison
    # We need to match [n_points, n_windows, n_models, window_size]
    model_train_loss_checkem = model_train_loss_checkem.expand(
        n_points, -1, n_models, -1) # [n_points, n_windows, n_models, window_size]


    # 2. Apply Constraint Check (L_ref(w_ref) > L_ref(w) - gamma_slack)
    boundary_passed = torch.all(model_train_loss_checkem > (train_loss_checkem - gamma_slack), dim=-1)
    
    # Set the original model (first sample) to always pass the boundary
    # Note: If window_size > 1, this needs careful indexing, but for window_size=1 it's just index 0.
    boundary_passed[:, 0, :] = True 
    # [n_points, n_windows, n_models]

    # 3. Find Minimum Objective Loss (within boundary)
    # n_windows is already calculated implicitly by the shape of boundary_passed
    n_windows = boundary_passed.shape[1]
    
    # obj_loss is now [B, S, M]. We crop it to the correct number of windows.
    opt_checkem = obj_loss[:, :n_windows, :]
    
    # Discard samples outside the boundary
    opt_checkem[~boundary_passed] = torch.inf 
    
    # Find index of the sample that minimizes the objective loss
    # pick_samples is index *within the windows*
    pick_samples_relative = torch.min(opt_checkem, dim=1)[1] # [n_points, n_models]

    # 4. Extract the Predictions for the Best Sample
    # Output buffer is [n_points, 1, n_models, C, (H, W)]
    
    # Create a temporary tensor to hold the best prediction from the trajectory
    # The output should have shape [B, 1, M, ...]
    buffer_shape = [n_points, 1, n_models] + list(pred_dims)
    buffer = torch.zeros(buffer_shape, dtype=sample_preds.dtype, device=sample_preds.device)

    # Vectorization requires gathering using indices.
    # The index we want is pick_samples_relative + (index of the window) which is not trivial.
    # Sticking to the loop is simpler and correct given the index calculation.
    for i in range(n_points):
        for j in range(n_models):
            # pick_samples_relative[i, j] is the index of the best sample *within the windowed view*.
            # For window_size=1, this is the index directly into the original S dimension.
            sample_index = pick_samples_relative[i, j].item()
            buffer[i, 0, j] = sample_preds[i, sample_index, j]

    return {
        'average_net_pred': average_net_pred,
        'sample_preds': buffer,
    }

@torch.no_grad()
def combine_sample_last(
    average_net_pred: torch.Tensor, # [n_points, n_output_features]
    sample_preds: torch.Tensor, # [n_points, n_samples, n_models, n_output_features]
    train_loss, # [n_points, n_samples, n_models]
    **kwargs
):
    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds[:, -1, :, :].unsqueeze(1),
    }



@torch.no_grad()
def combine_sample_softmax(average_net_pred, sample_preds, train_loss, temperature=1., num_samples=None, **kwargs):
    # sample_preds: [B, S, M, C, H, W]
    # train_loss  : [B, S, M] or [B, S, M, C]
#     print('train_loss',train_loss.shape)
    if num_samples is None:
        num_samples = train_loss.shape[1]  # use S

    sample_preds = sample_preds[:, :num_samples, :, :, :, :]
    train_loss   = train_loss[:,  :num_samples, ...]

    # Reduce class dim if present -> [B, S, M]
    if train_loss.ndim == 4:
        train_loss = train_loss.mean(dim=-1)

    # Convert losses to weights via softmax over the S*M trajectory
    B, S, M = train_loss.shape
    sample_weights = torch.softmax(-train_loss.view(B, -1) / temperature, dim=-1).view(B, S, M)

    return {
        'average_net_pred': average_net_pred,  # [B, C, H, W]
        'sample_preds': sample_preds,          # [B, S, M, C, H, W]
        'sample_weights': sample_weights       # [B, S, M]
    }


@torch.no_grad()
def combine_sample_all(
    average_net_pred: torch.Tensor, # [n_points, n_output_features]
    sample_preds: torch.Tensor, # [n_points, n_samples, n_models, n_output_features]
    **kwargs
):
    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds,
    }




@torch.no_grad()
def combine_sample_weighted_all(
    average_net_pred: torch.Tensor, # [n_points, n_output_features]
    sample_preds: torch.Tensor, # [n_points, n_samples, n_models, n_output_features]
    train_loss: torch.Tensor, # [n_points, n_samples, n_models]
    temperature: float = 1.,
    **kwargs
):
    '''
    Passes ALL samples through (like combine_sample_all) but also computes
    the softmax weights based on the training loss (like combine_sample_softmax).
    This enables MIS calculation in downstream functions.
    '''
    
    if train_loss.ndim == 4:
        train_loss = train_loss.mean(dim=-1)
    B, S, M = train_loss.shape


    # Convert losses to weights via softmax over the S*M trajectory (MIS Weights)
    sample_weights = torch.softmax(-train_loss.view(B, -1) / temperature, dim=-1).view(B, S, M)

    return {
        'average_net_pred': average_net_pred,
        'sample_preds': sample_preds,
        'sample_weights': sample_weights  # The key addition for MIS
    }



@torch.no_grad()
def calculate_uncertainty_setting_b(
    average_net_pred: torch.Tensor,
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''
    Calculates Uncertainty using Setting (b) - Reference model is pre-selected (average_net_pred).
    This computes the Expected Cross-Entropy (ECE) decomposition.
    
    Inputs:
        average_net_pred: [B, C, H, W] (Segmentation) or [B, C] (Classification)
        sample_preds: [B, S, M, C, H, W] (Segmentation) or [B, S, M, C] (Classification)
        sample_weights: [B, S, M]
    '''
    eps = 1e-8
    
    # 1. Standardize Dimensions to [B, C, H, W] and [B, S, M, C, H, W]
    if average_net_pred.ndim == 2:
        # Classification case: [B, C] -> [B, C, 1, 1]
        average_net_pred = average_net_pred.unsqueeze(-1).unsqueeze(-1)
        sample_preds_standard = sample_preds.unsqueeze(-1).unsqueeze(-1) # [B, S, M, C, 1, 1]
        is_classification = True
    else:
        # Segmentation case: [B, C, H, W] (already correct)
        sample_preds_standard = sample_preds
        is_classification = False
    
    # Reshaping: Flatten S and M dimensions together: [B, S*M, C, H, W]
    sample_preds_merged = sample_preds_standard.flatten(1, 2)
    BMA_pred = average_net_pred # The reference prediction p(y|x, D)

    if sample_weights is None:
        # Simple unweighted average (Not MIS)
        
        # Calculate CE[BMA || w_n]: Sum over classes (dim=2). Result: [B, S*M, H, W]
        ce_per_sample = - torch.sum(
            (BMA_pred.unsqueeze(1) * torch.log(sample_preds_merged + gamma)), dim=2)
            
        # Total = E_w [CE[BMA || w]] approx (1/N) * sum_n( CE[BMA || w_n] )
        total = torch.mean(ce_per_sample, dim=1) # Average over S*M
        
        # Aleatoric = H[BMA]: [B, H, W]
        aleatoric = - torch.sum( (BMA_pred+gamma) * torch.log(BMA_pred+gamma), dim=1)
        epistemic = total - aleatoric
    else:
        # MIS Weighted Summation
        # sample_weights is [B, S, M]
        sample_weights_merged = sample_weights.flatten(1, 2) # [B, S*M]
        
        # *** FIX START ***
        # Ensure sample_preds_merged is truncated to match the number of weights (N = S*M)
        # This handles cases where the samples were combined with a longer trajectory than the weights.
        
        N_weights = sample_weights_merged.shape[1]
        N_preds = sample_preds_merged.shape[1]

        if N_weights != N_preds:
             print(f"Warning: Sample predictions ({N_preds}) and weights ({N_weights}) mismatch. Truncating samples.")
             sample_preds_merged = sample_preds_merged[:, :N_weights, ...]
        
        # Calculate Cross-Entropy (CE) per sample: CE[BMA || w_n]
        # Sum over classes (dim=2): result [B, S*M, H, W]
        ce_per_sample = - torch.sum(
             (BMA_pred.unsqueeze(1) * torch.log(sample_preds_merged + gamma)), dim=2)
        
        # Total = Weighted Average of CE (MIS Estimate): sum_n (Weight_n * CE_n)
        # sample_weights_merged: [B, S*M] -> unsqueeze twice to [B, S*M, 1, 1] for spatial broadcasting
        total = torch.sum(
            ce_per_sample * sample_weights_merged.unsqueeze(-1).unsqueeze(-1), 
            dim=1) # Sum over S*M, result [B, H, W]
        
        # Aleatoric = H[BMA]
        aleatoric = - torch.sum( (BMA_pred+gamma) * torch.log(BMA_pred+gamma), dim=1) # [B, H, W]
        
        epistemic = total - aleatoric

    # Squeeze out spatial dimensions if they were singletons (classification case)
    if is_classification:
        return {
            'total': total.squeeze(),
            'aleatoric': aleatoric.squeeze(),
            'epistemic': epistemic.squeeze(),
        }

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }


def calculate_uncertainty_setting_a(
    average_net_pred: torch.Tensor, # Note: This is ignored in MIS/MC Setting A
    sample_preds: torch.Tensor,
    sample_weights: torch.Tensor = None,
    gamma=1e-10,
    **kwargs
):
    '''
    Calculates Uncertainty using Setting (a) - Reference model is the BMA estimated from samples.
    This computes the Expected Entropy (EE) decomposition.
    
    Inputs:
        average_net_pred: [B, C, H, W] or [B, C] (Ignored in this setting)
        sample_preds: [B, S, M, C, H, W] (Segmentation) or [B, S, M, C] (Classification)
        sample_weights: [B, S, M]
    '''
    eps = 1e-8
    sample_preds = torch.clamp(sample_preds, eps, 1.0)
    
    # 1. Standardize Dimensions
    if sample_preds.ndim == 4: # [B, S, M, C]
        # Classification case: [B, S, M, C] -> [B, S, M, C, 1, 1]
        sample_preds_standard = sample_preds.unsqueeze(-1).unsqueeze(-1)
        is_classification = True
    else:
        # Segmentation case: [B, S, M, C, H, W]
        sample_preds_standard = sample_preds
        is_classification = False


    # Reshaping: Flatten S and M dimensions together: [B, total_samples, C, H, W]
    B = sample_preds_standard.shape[0]
    sample_preds_merged = sample_preds_standard.flatten(1, 2)
    
    # 2. HANDLE WEIGHTS (MIS)
    if sample_weights is not None:
        sample_weights = sample_weights.flatten(1, 2)  # [B, total_samples]
        
        # --- FIX START ---
        # Ensure sample_preds_merged is truncated to match the number of weights (N = S*M)
        N_weights = sample_weights.shape[1]
        N_preds = sample_preds_merged.shape[1]

        if N_weights != N_preds:
             print(f"Warning: Sample predictions ({N_preds}) and weights ({N_weights}) mismatch. Truncating samples.")
             sample_preds_merged = sample_preds_merged[:, :N_weights, ...]
        # --- FIX END ---
        
        # Normalize weights (critical for correct averaging)
        sample_weights = sample_weights / (sample_weights.sum(dim=1, keepdim=True) + eps) 
        
        # a) Weighted mean prediction (BMA estimate): E_w [p(y|x, w)]
        # sample_weights: [B, N] -> [B, N, 1, 1, 1]
        weighted_preds = sample_preds_merged * sample_weights[:, :, None, None, None]  # broadcasting
        mean_pred = weighted_preds.sum(dim=1)  # [B, C, H, W]

        # b) Total uncertainty: Entropy of BMA (H[E_w [p(y|x, w)]])
        # Sum over classes (dim=1): result [B, H, W]
        total = - torch.sum(mean_pred * torch.log(mean_pred), dim=1)

        # c) Aleatoric uncertainty: Weighted average entropy (E_w [H[w]])
        # H[w]: [B, N, H, W]
        entropy_per_sample = - torch.sum(sample_preds_merged * torch.log(sample_preds_merged), dim=2)
        # weighted_entropy: [B, N, H, W] * [B, N, 1, 1] -> [B, N, H, W]
        weighted_entropy = entropy_per_sample * sample_weights[:, :, None, None]
        # Sum over samples (dim=1): result [B, H, W]
        aleatoric = weighted_entropy.sum(dim=1)

    # 3. HANDLE UNWEIGHTED (Simple MC or Unweighted Ensemble)
    else:
        # a) Unweighted mean prediction (Simple BMA approximation)
        mean_pred = torch.mean(sample_preds_merged, dim=1)  # [B, C, H, W]
        
        # b) Total uncertainty: Entropy of BMA
        total = - torch.sum(mean_pred * torch.log(mean_pred), dim=1)  # [B, H, W]
        
        # c) Aleatoric uncertainty: Unweighted average entropy
        # Sum over classes (dim=2): [B, S*M, H, W] -> Mean over samples (dim=1): [B, H, W]
        aleatoric = - torch.mean(torch.sum(sample_preds_merged * torch.log(sample_preds_merged), dim=2), dim=1)

    epistemic = total - aleatoric

    # Squeeze out spatial dimensions if they were singletons (classification case)
    if is_classification:
        return {
            'total': total.squeeze(),
            'aleatoric': aleatoric.squeeze(),
            'epistemic': epistemic.squeeze(),
        }

    return {
        'total': total,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
    }
