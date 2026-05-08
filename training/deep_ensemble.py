import os
import torch
import natsort


def load_unet_model_deep(model_class, model_args, model_path: str, num_models: int = 5, device="cuda"):
    """
    Load multiple models for deep ensemble.

    Args:
        model_class: model constructor (e.g. AttUNet)
        model_args: dict of arguments for model_class
        model_path: directory with model checkpoints
        num_models: number of ensemble models to load
        device: torch device

    Returns:
        models: list of loaded models
    """
    models = []
    ckpt_folders = natsort.natsorted(os.listdir(model_path))[:num_models]

    for subdir in ckpt_folders:
        net = model_class(**model_args)
        ckpt_dir = os.path.join(model_path, subdir)
        ckpt_file = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        net.load_state_dict(torch.load(ckpt_file, map_location=device))
        net.to(device)
        net.eval()
        models.append(net)

    return models


def deep_ensemble_predict(models, image):
    """
    Run deep ensemble inference.

    Args:
        models: list of trained models
        image: input tensor [B,C,H,W]

    Returns:
        predictions: stacked probabilities [num_models, B, C, H, W]
        predictions_max: argmax predictions [num_models, B, H, W]
    """
    preds = []
    for model in models:
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs)

    predictions = torch.stack(preds, dim=0)  # [num_models, B, C, H, W]
    predictions_max = torch.argmax(predictions, dim=2)  # [num_models, B, H, W]

    return predictions, predictions_max
