import copy
import numpy as np
import logging
import sys
import os 
from tqdm import tqdm
# import utils_unet as utils
import shutil
import natsort
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Callable, List, Dict, Union
from utils.evaluation import compute_dice_and_hd
from utils.visualization import  plot_segmentation_results_final
import torch

import pandas as pd
from pathlib import Path
import yaml

import uncertainty_metrics

from utils.seeding import fix_seeds
global_seed = 42
fix_seeds(global_seed)






def record_metrics(tag, uncert, sample_for_tag, name, masks, metrics_store,save_dir_partial,
                   include_classes=None, make_macro_row=True):
    """
    Multiclass-aware recorder.
    - Calls evaluate_uncertainty_metrics once per target class.
    - Appends one row per class to metrics_store[tag].
    - Optionally appends a macro-average row across chosen classes.

    Args:
      include_classes: list/tuple of class ids to evaluate.
                       If None -> inferred as 1..C-1 (skip background=0).
      make_macro_row: add a single macro row averaged over include_classes.
    """
    # Infer number of classes C from predictions or masks
    # sample_for_tag["average_net_pred"]: (B, C, H, W)
    if "average_net_pred" in sample_for_tag and sample_for_tag["average_net_pred"].ndim >= 4:
        C = int(sample_for_tag["average_net_pred"].shape[1])
    else:
        # fallback: infer from masks max label
        C = int(masks.max().item()) + 1

    # default: skip background (0), evaluate foreground classes
    if include_classes is None:
        include_classes = list(range(1, C))
    else:
        include_classes = list(include_classes)

    metrics_store.setdefault(tag, [])

    per_class_rows = []
    for k in include_classes:
        case_metrics = uncertainty_metrics.evaluate_uncertainty_metrics(
            tag = tag,
            name        = name[0] if isinstance(name, (list, tuple)) else name,
            masks       = masks,
            uncert_best = uncert,
            sample      = sample_for_tag,
            target_class= k,
            save_dir = save_dir_partial
        )
        
#         print('case_metrics', case_metrics)
        row = dict(case_metrics)
        row["variant"] = tag
        row["class_id"] = k
        metrics_store[tag].append(row)
        per_class_rows.append(row)
#         print(f"[metrics::{tag}::class={k}] {case_metrics}")

    # Optional: add a macro-average (across include_classes) row
    if make_macro_row and per_class_rows:
        # choose numeric keys to average
        numeric_keys = [k for k, v in per_class_rows[0].items() if isinstance(v, (int, float))]
        macro = {nk: float(np.mean([r[nk] for r in per_class_rows])) for nk in numeric_keys}
        macro.update({
            "name": per_class_rows[0]["name"],
            "variant": tag,
            "class_id": "macro(" + ",".join(map(str, include_classes)) + ")"
        })
        metrics_store[tag].append(macro)
#         print(f"[metrics::{tag}::macro over {include_classes}] {macro}")


















def gather_quality_metrics(tag, test_images, uncert, comb, masks, img_name, metrics_store,
                           compute_dice_and_hd_fn, save_dir, img_h=224, img_w=224):
    """Collect one row of metrics and append to metrics_store[tag]."""
    epi_mean  = uncert["epistemic"][0].mean().item()
    alea_mean = uncert["aleatoric"][0].mean().item()
    tot_mean  = uncert["total"][0].mean().item()

    # comb['sample_preds'] shape example: [1, M, S, C, H, W]
    shp = comb['sample_preds'].shape
    assert len(shp) >= 5, f"Unexpected shape for sample_preds: {shp}"
    C = shp[-3]
    H = shp[-2]; W = shp[-1]
    n_models  = shp[1]

    sample_preds_reshaped = comb['sample_preds'].view(
        1, n_models * shp[2], C, H, W
    )
    avg_pred_expanded = comb['average_net_pred'].unsqueeze(1)  # [1,1,C,H,W]
    all_preds = torch.cat([sample_preds_reshaped, avg_pred_expanded], dim=1)[0].cpu().numpy()  # [num_samples,C,H,W]

    # NOTE: masks[0] is expected [A,H,W] with multiclass ints (0..C-1)
    scores = compute_dice_and_hd_fn(all_preds, masks[0])
#     print('scores', scores)
    plot_segmentation_results_final(
        test_images, masks, scores,
        uncert["epistemic"], uncert["aleatoric"], uncert["total"],
        img_name, save_dir
    )

    metrics_store[tag].append({
        "Image_Name"    : img_name[0],
        "epistemic_mean": epi_mean,
        "aleatoric_mean": alea_mean,
        "total_mean"    : tot_mean,
        "dice_mean"     : scores["Dice_Statistics"]["mean"],          # samples vs mean (macro Dice)
        "hd_mean"       : scores["Hausdorff_Distance_Statistics"]["mean"],  # samples vs mean (macro HD)
        "gt_dice"       : scores["GT_Dice"],                          # mean vs consensus (macro Dice)
    })
