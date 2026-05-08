import os
import torch
from typing import Iterable
import numpy as np
import random

# def fix_seeds(seed: int = 42):
#     # for reproducibility on GPU with cudatoolkit >= 10.2
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#     # reproducibility
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)

#     random.seed(seed)
#     np.random.seed(seed)


def fix_seeds(seed: int = 42):
    # Set environment variable required for some cuBLAS versions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Seed python, numpy, and torch random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False