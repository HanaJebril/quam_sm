import logging, os, torch
from training.adversarial_training import generate_adversarial_models_seg
from evaluation_pipeline import run_pipeline_on_dataset
from utils.seeding import fix_seeds
from utils.loaders import FundusBmpDataset
from config import load_config
from models.attentionunet import AttUNet
from torch.utils.data import Dataset, DataLoader
import natsort


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Fix randomness
    fix_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load config
    config = load_config("configs/config_quam_atten_unet_erosion_dilation.yaml")
    # Load pretrained reference model
    ref_model = AttUNet(in_channels=config['model']['in_channels'], n_classes=config['model']['n_classes'], is_batchnorm= False, drop=config['model']['drop_ref'], channels=config['model']['channels'])
    
    bestmodelpath_ref = os.path.join(config['model']['model_path'],
                                     natsort.natsorted(os.listdir(config['model']['model_path']))[-1])

    restore_path_ref = os.path.join(config['model']['model_path'],
                                natsort.natsorted(os.listdir(config['model']['model_path']))[-1]) + '/' + \
                   os.listdir(bestmodelpath_ref)[0]
    
    
    ref_model.load_state_dict(torch.load(restore_path_ref))
    ref_model = ref_model.to(device=device);

    
    
    # Load pretrained MC model
    mc_model = AttUNet(in_channels=config['model']['in_channels'], n_classes=config['model']['n_classes'], is_batchnorm= False, drop=config['model']['drop_mc'], channels=config['model']['channels'])
    
    bestmodelpath_mc = os.path.join(config['model']['mc_model_path'],
                                     natsort.natsorted(os.listdir(config['model']['mc_model_path']))[-1])
    restore_path_mc = os.path.join(config['model']['mc_model_path'],
                                natsort.natsorted(os.listdir(config['model']['mc_model_path']))[-1]) + '/' + \
                   os.listdir(bestmodelpath_mc)[0]
    
    
    mc_model.load_state_dict(torch.load(restore_path_mc))
    mc_model = mc_model.to(device=device);
    
    
    # Load train set
    train_ds = FundusBmpDataset(config['train']['path'], size=config['train']['data_size'], augment=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=4)

    # Run pipeline
    for dataset_config in config['testing']:
#         run_pipeline_on_dataset(dataset_config, train_loader, ref_model, ...)
        run_pipeline_on_dataset(dataset_config, train_loader, ref_model, mc_model, config['model']['model_path'],  config['model']['penalties_per_optimisation_step'])
