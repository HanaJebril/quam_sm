import os
import torch
from tqdm import tqdm
from training.adversarial_training import generate_adversarial_models_seg
from utils.metrics import record_metrics, gather_quality_metrics
from utils.visualization import (
    plot_model_outputs, plt_uncertainty, plot_mc_outputs
)
from utils.scheduling import configure_penopt_schedule
from inference import mc_dropout, ensemble, tta
import uncertainty_quam
from utils.loaders import FundusBmpDataset
import pandas as pd
from pathlib import Path
import natsort
from utils.evaluation import compute_dice_and_hd    
    
def run_pipeline_on_dataset(config, train_loader, ref_model, ref_model_mc, model_path, penalties_per_optimisation_step):
    """
    Run QUAM + MC dropout + deep ensemble uncertainty evaluation on one dataset.

    Args:
        config (dict): dataset config (data_path, output_folder, batch_size, etc.)
        train_loader: DataLoader for training set
        ref_model: reference model (without dropout)
        ref_model_mc: reference model with dropout enabled
        model_path (str): path to deep ensemble checkpoints
    """
    

    # === Load dataset ===
    test_dataset = FundusBmpDataset(
        root_dir=config['data_path'],
        size=config['data_size'],
        augment=config['transform'],
        test=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=4
    )

    # === Setup ===
    save_dir_base = config['output_folder']
    os.makedirs(save_dir_base, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics_store_uncert = {v: [] for v in ["mis","best", "all","all_a_weighted", "last", "mc", "softmax", "deep", "tta"]}
    metrics_store_quality = {v: [] for v in ["mis","best", "all", "last", "mc", "softmax", "deep", "tta"]}

    # === Loop over test images ===
    for test_image, masks, name in tqdm(test_loader, desc="Evaluating"):
        save_dir = os.path.join(save_dir_base, name[0])
        os.makedirs(save_dir, exist_ok=True)
        
        
        c_scheduling = configure_penopt_schedule(
            schedule_type='lin',        # linear schedule
            c0=1.0,            # starting value of c
            eta=2.0,           # how much to increase c every `update_c_every` steps
            update_c_every=3  # how often to update c
        )
        penalties_per_optimisation_step = 50
        # QUAM adversarial models
        sample = generate_adversarial_models_seg(
            ref_model, train_loader, test_image,
            num_models=2,
            initial_c=1.0,
            c_scheduling = c_scheduling, 
            num_epochs=3,
            penalties_per_optimisation_step=penalties_per_optimisation_step,
            gamma=0.05,
            lr=5e-4,
            device=device,
            save_dir=save_dir
        )
        ref_model.eval()
        n_penalties = len(train_loader) // penalties_per_optimisation_step
        plot_model_outputs(
            ref_pred=sample['average_net_pred'],
            adv_preds=sample['sample_preds'],
            test_image=test_image,
            save_dir=save_dir,
            filename="segmentation_outputs_quam.png",
            class_index=1
        )
        
        
        # quam -- mis -- b
        save_dir_partial = os.path.join(save_dir, 'mis');
        os.makedirs(save_dir_partial, exist_ok=True)
    

        uncer_mis = uncertainty_quam.combine_sample_mis_weighted(
            sample['average_net_pred'],
            sample['sample_preds'], 
            sample['train_loss']
        )


        # quam -- mis -- a
        
        uncert_mis_a = uncertainty_quam.calculate_uncertainty_setting_a(
            average_net_pred = uncer_mis['average_net_pred'],
            sample_preds     = uncer_mis['sample_preds'],
            sample_weights   = uncer_mis['sample_weights']
        )
        plt_uncertainty(uncert_mis_a, test_image, 'mis_a', save_dir_partial)
        record_metrics("mis_a", uncert_mis_a, uncer_mis, name, masks, metrics_store_uncert,save_dir_partial, include_classes=[1], make_macro_row=False)  
        
        gather_quality_metrics("mis", test_image, uncert_mis_a, uncer_mis,
                        masks, name, metrics_store_quality,
                        compute_dice_and_hd, save_dir_partial)
        
        
        
    # === Save results ===

    for tag, rows in metrics_store_uncert.items():
        if not rows: continue
        df = pd.DataFrame(rows)
        out_dir = Path(save_dir_base) / f"{tag}_uncert"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "uncert_results.csv", index=False)

    for tag, rows in metrics_store_quality.items():
        if not rows: continue
        df = pd.DataFrame(rows)
        out_dir = Path(save_dir_base) / f"{tag}_quality"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "quality_results.csv", index=False)
