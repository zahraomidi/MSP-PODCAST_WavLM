import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
import os

def set_global_seed(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

def _hp_get(hparams, key, default=None):
    """Safe getter for both attr-style and dict-style hparams."""
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)