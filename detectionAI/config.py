import os
import random
from pathlib import Path

import numpy as np
import torch

class Config():
    ###
    #   Change your settings
    ###
    detection_root_path = str(Path(__file__).parent)
    checkpoint_dir = os.path.join(detection_root_path, "checkpoints")
    checkpoint_path = os.path.join(detection_root_path, "checkpoints/best.ckpt")
    training_dir = os.path.join(detection_root_path, "data/train/")
    testing_dir = os.path.join(detection_root_path, "data/test")
    ref_path = os.path.join(detection_root_path, "data/label.json")
    
    seed = 100

    train_data_ref_batch_size = 64
    train_data_iter = 3
    train_batch_size = 256
    train_batch_group_size = 1
    train_number_epochs = 1000
    imgsize = (27,41)
    
    thr_max_dist = 0.7
    
    num_workers = 6
    
    device = None
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')      # for OSX M1
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    @staticmethod
    def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        
        if Config.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

