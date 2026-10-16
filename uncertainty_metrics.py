import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import wasserstein_distance




def brier_score(p_pred, p_true):
    """Mean squared difference between predicted and annotator soft labels."""
    return float(np.mean((p_pred - p_true) ** 2))

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """ECE between predicted prob and observed soft label."""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_true = np.mean(y_true[mask])
            prob_pred = np.mean(y_pred[mask])
            ece += np.abs(prob_true - prob_pred) * np.sum(mask)
    return float(ece / len(y_true))

def jensen_shannon_divergence(p, q, eps=1e-8):
    """Symmetric divergence between distributions."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm) / p.size)

def probabilistic_dice(p, q):
    """Soft Dice overlap."""
    num = 2 * np.sum(p * q)
    den = np.sum(p**2) + np.sum(q**2) + 1e-8
    return float(num / den)

def entropy_corr(p, q, eps=1e-8):
    """Correlation between model and annotator entropy maps."""
    def entropy(x):
        return -(x*np.log(x+eps) + (1-x)*np.log(1-x+eps))
    Hp, Hq = entropy(p), entropy(q)
    return float(np.corrcoef(Hp.flatten(), Hq.flatten())[0,1])

def net_benefit_curve(y_true_soft, y_pred, thresholds=np.linspace(0.01, 0.99, 99)):
    """Compute Net Benefit values across thresholds."""
    y_true_hard = (y_true_soft >= 0.5).astype(float)
    N = len(y_true_hard)
    NB = []
    for t in thresholds:
        pred_bin = (y_pred >= t).astype(float)
        TP = np.sum((pred_bin == 1) & (y_true_hard == 1))
        FP = np.sum((pred_bin == 1) & (y_true_hard == 0))
        nb = (TP / N) - (FP / N) * (t / (1 - t))
        NB.append(nb)
    return thresholds, np.array(NB)

def net_benefit_auc(y_true_soft, y_pred):
    thresholds, NB = net_benefit_curve(y_true_soft.flatten(), y_pred.flatten())
    return float(auc(thresholds, NB))












def earth_movers_distance(p, q):
    """
    Compute 1D Earth Mover's Distance (Wasserstein-1) between two flattened maps.
    Args:
        p, q: np.ndarray in [0,1], same shape (e.g., uncertainty maps)
    Returns:
        float: Earth Mover's Distance
    """
    p = p.flatten()
    q = q.flatten()
    return float(wasserstein_distance(p, q))












# -------- robust helpers --------
def normalize_01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def to_binary_boundary(mask):
    structure = np.ones((3,3), dtype=bool)
    er = binary_erosion(mask.astype(bool), structure=structure, border_value=0)
    return mask.astype(bool) ^ er

def distance_to_boundary(mask):
    bd = to_binary_boundary(mask)
    return distance_transform_edt(~bd)

def symmetric_hd95(pred_mask, true_mask):
    d_true = distance_to_boundary(true_mask)
    d_pred = distance_to_boundary(pred_mask)
    bd_true = to_binary_boundary(true_mask)
    bd_pred = to_binary_boundary(pred_mask)
    dt = d_true[bd_pred]
    dp = d_pred[bd_true]
    if dt.size == 0 or dp.size == 0:
        return 0.0
    return max(np.percentile(dt, 95), np.percentile(dp, 95))

def make_error_map_binary(y_pred_hw, y_true_hw):
    """Both inputs [H,W] binary ints {0,1}."""
    return (y_pred_hw != y_true_hw).astype(np.float32)

def pcc_and_r2(a, b, eps=1e-8):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a_m = a.mean(); b_m = b.mean()
    cov = ((a - a_m) * (b - b_m)).mean()
    std_a = a.std() + eps
    std_b = b.std() + eps
    pcc = cov / (std_a * std_b)
    r2 = pcc**2
    
    
    # Flatten for metric computations
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(pcc), float(r2), brier_score(a_flat,b_flat), expected_calibration_error(a_flat,b_flat),entropy_corr(a_flat,b_flat), probabilistic_dice(a_flat,b_flat), earth_movers_distance(a_flat,b_flat)




def uce(U, E, n_bins=15):
    U = U.ravel()
    E = E.ravel()
    assert U.shape == E.shape, f"UCE expects same number of voxels, got U={U.shape}, E={E.shape}"
    bins = np.linspace(0, 1, n_bins+1)
    inds = np.digitize(U, bins) - 1
    out = 0.0; N = len(U)
    for b in range(n_bins):
        idx = (inds == b)
        nb = int(idx.sum())
        if nb == 0: 
            continue
        mu_u = float(U[idx].mean())
        mu_e = float(E[idx].mean())
        out += (nb / N) * abs(mu_u - mu_e)
    return float(out)

def aurc(U, E):
    u = U.ravel(); e = E.ravel()
    assert u.shape == e.shape, f"AURC expects same number of voxels, got U={u.shape}, E={e.shape}"
    order = np.argsort(-u)
    e_sorted = e[order]
    N = len(e_sorted)
    cum_err = np.cumsum(e_sorted)
    total_err = float(cum_err[-1]) if N>0 else 0.0
    kept = np.arange(N, 0, -1)
    risks = (total_err - np.concatenate([[0], cum_err[:-1]])) / kept
    coverages = kept / N if N>0 else kept
    return float(np.trapz(risks[::-1], coverages[::-1])) if N>0 else 0.0

def buc(U, y_pred, y_true, use_hd95=True, tau_pixels=5):
    tau = symmetric_hd95(y_pred, y_true) if use_hd95 else float(tau_pixels)
    d_pred = distance_to_boundary(y_pred)
    R = d_pred <= tau
    notR = ~R
    mu_R = float(U[R].mean()) if R.any() else 0.0
    mu_notR = float(U[notR].mean()) if notR.any() else 0.0
    denom = mu_R + mu_notR
    return float(mu_R / denom) if denom > 0 else 0.0

def ba_ece(U, E, y_true, K=10, max_dist=None, eps=1e-8):
    d = distance_to_boundary(y_true)
    if max_dist is None:
        max_dist = np.percentile(d, 95)
    d = np.clip(d, 0, max_dist)
    bins = np.linspace(0, max_dist, K+1)
    inds = np.digitize(d.ravel(), bins) - 1

    mu_u, mu_e, avg_dist = [], [], []
    Ur = U.ravel(); Er = E.ravel()
    for i in range(K):
        idx = (inds == i)
        if not np.any(idx):
            mu_u.append(0.0); mu_e.append(0.0); avg_dist.append(np.inf); continue
        mu_u.append(float(Ur[idx].mean()))
        mu_e.append(float(Er[idx].mean()))
        avg_dist.append(float(d.ravel()[idx].mean()))
    avg_dist = np.array(avg_dist, np.float32)
    inv = 1.0 / (avg_dist + eps)
    inv[np.isinf(avg_dist)] = 0.0
    w = inv / (inv.sum() + eps)
    mu_u = np.array(mu_u); mu_e = np.array(mu_e)
    return float(np.sum(w * np.abs(mu_u - mu_e)))

def space(U, E, sigma=2.0):
    U_s = gaussian_filter(U.astype(np.float32), sigma=sigma)
    E_s = gaussian_filter(E.astype(np.float32), sigma=sigma)
    return float(np.mean(np.abs(U_s - E_s)))

def to_hw_binary_mask(pred, target_class=1):
    """
    Convert prediction to a binary [H,W] numpy mask {0,1} for a given target class.
    Handles shapes with arbitrary leading dims (batch/models/samples), with or without a class dim.

    Assumptions:
      - Spatial dims are always the last two dims: [..., H, W]
      - If there is a class dimension, it is at index -3 (i.e., before H,W)
        with size >= 2 (e.g., C=3 for classes {0,1,2}).
      - All other leading dims (batch/ensembles/samples) are averaged.
    """
    import torch
    import numpy as np

    x = pred
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().cpu().float()

    if x.ndim < 2:
        raise ValueError(f"Expected at least 2 dims ending with [H,W], got {tuple(x.shape)}")

    H, W = x.shape[-2], x.shape[-1]

    # Detect class dim at -3 if present (C >= 2). Otherwise, treat as binary/prob map.
    has_class = (x.ndim >= 3) and (x.shape[-3] >= 2)

    # Reduce/average all leading dims except the class dim (if present)
    reduce_dims = []
    for i in range(0, x.ndim - 2):  # all non-spatial dims
        if has_class and i == (x.ndim - 3):
            continue  # keep class dim
        reduce_dims.append(i)

    if len(reduce_dims) > 0:
        x = x.mean(dim=reduce_dims, keepdim=False)

    if has_class:
        # x is now [C,H,W] → class labels
        labels = torch.argmax(x, dim=0)  # [H,W]
        hw = (labels == int(target_class)).to(torch.uint8).numpy()
    else:
        # x is now [H,W] (binary/prob) or [1,H,W]
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim != 2:
            raise ValueError(f"After reduction, expected [H,W], got {tuple(x.shape)}")
        hw = (x > 0.5).to(torch.uint8).numpy()

    return hw






# def plot_all_uncertainties(U_epi, U_ale, U_tot, U_anno,anno_var, name, save_path=None):
#     fig, axes = plt.subplots(1, 5, figsize=(16,4))

#     ims = []
#     ims.append(axes[0].imshow(U_epi, cmap="viridis"))
#     axes[0].set_title("Epistemic")
#     ims.append(axes[1].imshow(U_ale, cmap="viridis"))
#     axes[1].set_title("Aleatoric")
#     ims.append(axes[2].imshow(U_tot, cmap="viridis"))
#     axes[2].set_title("Total")
#     ims.append(axes[3].imshow(U_anno, cmap="viridis"))
#     axes[3].set_title("GT annotation entropy ")
#     ims.append(axes[4].imshow(anno_var, cmap="viridis"))
#     axes[4].set_title("GT annotation variance")

#     for ax in axes:
#         ax.axis("off")

#     # Add one shared colorbar
#     cbar = fig.colorbar(ims[-1], ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
#     cbar.set_label("Uncertainty (0–1)")

#     fig.suptitle(f"Uncertainty maps - {name}", fontsize=14)

#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(save_path, dpi=200)
#         plt.close()
#     else:
#         plt.show()

def plot_all_uncertainties(U_epi, U_ale, U_tot, U_anno, anno_var, name, save_path=None, normalize="fixed"):
    """
    Plot 5 uncertainty maps side by side with shared colorbar.

    Args:
        U_epi, U_ale, U_tot, U_anno, anno_var: 2D numpy arrays [H,W]
        name (str): case name
        save_path (str or None): if set, save to path instead of showing
        normalize (str): 
            - "fixed" → force [0,1] scale
            - "global" → scale using min/max across all maps
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Choose normalization ---
    if normalize == "fixed":
        vmin, vmax = 0.0, 1.0
    elif normalize == "global":
        vmin = min(np.min(arr) for arr in [U_epi, U_ale, U_tot, U_anno, anno_var])
        vmax = max(np.max(arr) for arr in [U_epi, U_ale, U_tot, U_anno, anno_var])
    else:
        raise ValueError("normalize must be 'fixed' or 'global'")

    fig, axes = plt.subplots(1, 5, figsize=(20,4))

    ims = []
    ims.append(axes[0].imshow(U_epi, cmap="viridis", vmin=vmin, vmax=vmax))
    axes[0].set_title("Epistemic")
    ims.append(axes[1].imshow(U_ale, cmap="viridis", vmin=vmin, vmax=vmax))
    axes[1].set_title("Aleatoric")
    ims.append(axes[2].imshow(U_tot, cmap="viridis", vmin=vmin, vmax=vmax))
    axes[2].set_title("Total")
    ims.append(axes[3].imshow(U_anno, cmap="viridis", vmin=vmin, vmax=vmax))
    axes[3].set_title("GT annotation entropy")
    ims.append(axes[4].imshow(anno_var, cmap="viridis", vmin=vmin, vmax=vmax))
    axes[4].set_title("GT annotation variance")

    for ax in axes:
        ax.axis("off")

#     # ✅ Shared colorbar across all
#     cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.04)
#     cbar.set_label("Uncertainty")

    fig.suptitle(f"Uncertainty maps - {name}", fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()






def evaluate_uncertainty_metrics(tag,name, masks, uncert_best, sample, target_class, save_dir):
    """
    masks: typically [1, A, H, W] (A annotators) or sometimes multiclass integer masks.
           Works with:
             - Binary per-annotator: values in {0,1}
             - Multiclass per-annotator (stored as ints): values in {0,1,2,...}
    uncert_best: dict with 'epistemic','aleatoric','total' each [1,H,W] (float)
    sample['sample_preds']: prediction with arbitrary leading dims and optional class dim at -3.
    """

    # ----- 1) Annotator uncertainty for target_class -----
    m = masks.squeeze(0)  # drop leading batch if present
    # Possible shapes now:
    #  [A,H,W] (binary or multiclass ints per annotator)
    #  [H,W]   (single mask, binary or multiclass ints)

    # Normalize to float for operations
    m_float = m.float()

    # Heuristic: treat as multiclass if values exceed 1 anywhere
    # (works for {0,1,2} etc.). Avoid .unique() on huge tensors for speed.
    is_multiclass = (m_float.max() > 1)

    if m_float.ndim == 3:
        # [A,H,W]
        if is_multiclass:
            # One-vs-rest indicator for the target class per annotator
            # Compare against int target_class on original (non-float) tensor to be safe
            m_ind = (m == int(target_class)).float()  # [A,H,W]
            p_anno = m_ind.mean(dim=0).cpu().numpy()  # [H,W] in [0,1]
        else:
            # Binary annotators already in {0,1}
            m_bin = (m_float > 0.5).float()
            p_anno = m_bin.mean(dim=0).cpu().numpy()
    elif m_float.ndim == 2:
        # [H,W] single mask
        if is_multiclass:
            p_anno = (m == int(target_class)).float().cpu().numpy()
        else:
            p_anno = (m_float > 0.5).float().cpu().numpy()
    else:
        raise ValueError(f"Unexpected masks shape after squeeze: {tuple(m.shape)}")

    eps = 1e-6
    p_safe = np.clip(p_anno, eps, 1.0 - eps)
    anno_ent = -(p_safe*np.log(p_safe) + (1-p_safe)*np.log(1-p_safe)) / np.log(2.0)  # [0,1]
    anno_var = (p_anno * (1.0 - p_anno)) / 0.25                                      # [0,1]
    U_anno = anno_ent  # or anno_var

    # Binary ground-truth for the chosen class via majority vote
    y_true = (p_anno >= 0.5).astype(np.uint8)  # [H,W]

    # ----- 2) Predictions → [H,W] binary for target_class -----
    y_pred = to_hw_binary_mask(sample['sample_preds'], target_class=target_class)  # [H,W], {0,1}

    # ----- 3) Model uncertainties → [H,W] in [0,1] -----
    U_epi = normalize_01(uncert_best["epistemic"].squeeze(0).detach().cpu().numpy())
    U_ale = normalize_01(uncert_best["aleatoric"].squeeze(0).detach().cpu().numpy())
    U_tot = normalize_01(uncert_best["total"].squeeze(0).detach().cpu().numpy())

    # ----- 4) Error map [H,W] -----
    H, W = U_tot.shape
    assert y_true.shape == (H, W) and y_pred.shape == (H, W), \
        f"Shapes must match: U={(H,W)}, y_true={y_true.shape}, y_pred={y_pred.shape}"
    E = make_error_map_binary(y_pred, y_true)

    # ----- 5) Correlations w/ annotator uncertainty -----
    pcc_epi, r2_epi,bri_epi, ece_epi,ent_cor_epi, brob_dice_epi, EMD_epi = pcc_and_r2(U_epi, U_anno)
    pcc_ale, r2_ale,bri_ale, ece_ale,ent_cor_ale, brob_dice_ale, EMD_ale = pcc_and_r2(U_ale, U_anno)
    pcc_tot, r2_tot,bri_tot, ece_tot,ent_cor_tot, brob_dice_tot, EMD_tot = pcc_and_r2(U_tot, U_anno)
    
    
    
    
#     plot_all_uncertainties(U_epi, U_ale, U_tot, U_anno,anno_var, name,
#                        save_path=f"{save_dir}/{name}_uncertainties.png")
    plot_all_uncertainties(U_epi, U_ale, U_tot, U_anno, anno_var, name=tag, save_path = f"{save_dir}/{name}_{tag}_uncertainties.png" , normalize="fixed")



    # ----- 6) Metrics -----
    def all_metrics(U):
        U = normalize_01(U)
        return {
            "UCE": uce(U, E),
            "AURC": aurc(U, E),
            "BUC": buc(U, y_pred, y_true, use_hd95=True),
            "BA-ECE": ba_ece(U, E, y_true, K=10),
            "SPACE": space(U, E, sigma=2.0),
        }

    mets_epi = all_metrics(U_epi)
    mets_ale = all_metrics(U_ale)
    mets_tot = all_metrics(U_tot)
    
    
    # ----- 7) Compare model predictive mean vs annotator soft labels -----
    # Get predictive mean probability map from sample (using your existing logic)
#     if 'average_net_pred' in sample and sample['average_net_pred'].ndim >= 4:
#         avg_pred = sample['average_net_pred'].mean(0)[target_class].cpu().numpy()  # [H,W]
#     else:
    # fallback: try from sample_preds (assuming MC samples)
    x = sample['sample_preds']
    if torch.is_tensor(x):
        x = x.detach().cpu().float()

    # Identify dimensions
    # Expect shape [..., C, H, W] where C>=2 if multiclass
    if x.ndim >= 3 and x.shape[-3] >= 2:
        # Average over all non-class, non-spatial dims
        reduce_dims = [i for i in range(x.ndim - 2) if i != (x.ndim - 3)]
        avg_pred = x.mean(dim=reduce_dims, keepdim=False)[target_class].numpy()
    else:
        # Binary or already single-channel case
        avg_pred = x.mean(dim=tuple(range(x.ndim - 2))).squeeze().numpy()


    avg_pred = normalize_01(avg_pred)
    p_anno_soft = p_anno  # [H,W]

    # Flatten for metric computations
    y_pred_flat = avg_pred.flatten()
    y_true_flat = p_anno_soft.flatten()

    brier = brier_score(y_pred_flat, y_true_flat)
    ece_val = expected_calibration_error(y_true_flat, y_pred_flat)
    jsd_val = jensen_shannon_divergence(p_anno_soft, avg_pred)
    dice_val = probabilistic_dice(p_anno_soft, avg_pred)
    ent_corr = entropy_corr(p_anno_soft, avg_pred)
    nb_auc_val = net_benefit_auc(p_anno_soft, avg_pred)


    return {
        "name": name,
        "corr_epi_pcc": pcc_epi, "corr_epi_r2": r2_epi,
        "corr_ale_pcc": pcc_ale, "corr_ale_r2": r2_ale,
        "corr_tot_pcc": pcc_tot, "corr_tot_r2": r2_tot,
        

        "BRIER_epi": bri_epi, "ECE_epi": ece_epi, "ENTR_CORR_epi": ent_cor_epi, "Prop_Dice_epi": brob_dice_epi, "EMD_epi": EMD_epi ,
        "BRIER_ale": bri_ale, "ECE_ale": ece_ale, "ENTR_CORR_ale": ent_cor_ale, "Prop_Dice_ale": brob_dice_ale, "EMD_ale": EMD_ale ,
        "BRIER_tot": bri_ale, "ECE_tot": ece_tot, "ENTR_CORR_tot": ent_cor_tot, "Prop_Dice_tot": brob_dice_tot, "EMD_tot": EMD_tot ,

        "UCE_epi": mets_epi["UCE"], "AURC_epi": mets_epi["AURC"], "BUC_epi": mets_epi["BUC"], "BAECE_epi": mets_epi["BA-ECE"], "SPACE_epi": mets_epi["SPACE"],
        "UCE_ale": mets_ale["UCE"], "AURC_ale": mets_ale["AURC"], "BUC_ale": mets_ale["BUC"], "BAECE_ale": mets_ale["BA-ECE"], "SPACE_ale": mets_ale["SPACE"],
        "UCE_tot": mets_tot["UCE"], "AURC_tot": mets_tot["AURC"], "BUC_tot": mets_tot["BUC"], "BAECE_tot": mets_tot["BA-ECE"], "SPACE_tot": mets_tot["SPACE"],

        "BrierScore": brier,
        "ECE": ece_val,
        "JSD": jsd_val,
        "ProbDice": dice_val,
        "EntropyCorr": ent_corr,
        "NetBenefitAUC": nb_auc_val
    }

