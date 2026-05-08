import os
import torch
import natsort
from models.unet_dropout import UNet


def load_unet_model(model_details):
    """
    Load a single UNet model with specified details and return it in eval mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(
        in_channels=model_details['in_channels'],
        n_classes=model_details['n_classes'],
        is_batchnorm=True,
        drop=0.3,
        channels=model_details['channels']
    )

    bestmodelpath = os.path.join(model_details['model_path'], natsort.natsorted(os.listdir(model_details['model_path']))[-1])
    restore_path = os.path.join(bestmodelpath, os.listdir(bestmodelpath)[0])

    net.load_state_dict(torch.load(restore_path, map_location=device))
    net.to(device)
    net.eval()
    return net, device


def load_unet_model_deep(model_details):
    """
    Load multiple UNet models (deep ensemble) from the specified directory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    best_res = natsort.natsorted(os.listdir(model_details['model_path']))

    for subdir in best_res:
        net = UNet(
            in_channels=model_details['in_channels'],
            n_classes=model_details['n_classes'],
            is_batchnorm=True,
            drop=0.3,
            channels=model_details['channels']
        )

        bestmodelpath = os.path.join(model_details['model_path'], subdir)
        restore_path = os.path.join(bestmodelpath, os.listdir(bestmodelpath)[0])

        net.load_state_dict(torch.load(restore_path, map_location=device))
        net.to(device)
        models.append(net)

    return models, device