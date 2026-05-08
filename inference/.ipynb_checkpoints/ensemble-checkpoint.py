import torch
import torch.nn.functional as F
from typing import List
import os
import torch
from pathlib import Path
import natsort
from models.attentionunet import AttUNet


def deep_ensemble_predict(models: List[torch.nn.Module], x: torch.Tensor):
    """
    Perform prediction using deep ensemble.

    Args:
        models (List[nn.Module]): List of trained models.
        x (torch.Tensor): Input tensor (B, C, H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Stacked probabilities (num_models, B, C, H, W)
            - Stacked argmax predictions (num_models, B, H, W)
    """
    predictions = []
    predictions_max = []

    for model in models:
        model.eval()
        with torch.no_grad():
            
            output = model(x)
            prob = F.softmax(output, dim=1)
            pred_argmax = torch.argmax(output, dim=1)
            predictions.append(prob)
            predictions_max.append(pred_argmax)

    predictions = torch.cat(predictions, dim=0)
    predictions_max = torch.cat(predictions_max, dim=0)
    return predictions, predictions_max



def load_unet_model_deep(model_path):
    """
    Load multiple UNet models (deep ensemble) from the specified directory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    best_res = natsort.natsorted(os.listdir(model_path))[:5]

    for subdir in best_res:
        net = AttUNet(in_channels=3, n_classes=2,is_batchnorm= False, drop=0., channels=64);

        bestmodelpath = os.path.join(model_path, subdir)
        restore_path = os.path.join(bestmodelpath, os.listdir(bestmodelpath)[0])

        net.load_state_dict(torch.load(restore_path, map_location=device))
        net.to(device)
        models.append(net)

    return models, device    
        