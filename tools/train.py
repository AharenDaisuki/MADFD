import os
import time
import logging
import argparse
import warnings
import numpy 
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from models.MAT import MAT
# from datasets.dataset import DeepfakeDataset
from datasets.custom_dataset import MADFD_dataset
from AGDA import AGDA
import cv2
from utils import dist_average,ACC
assert torch.cuda.is_available()

def train(args):
    config = args.config
    
    seed = config.seed
    root = config.root
    batch_size = config.batch_size

    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(0) # single gpu training by default

    train_dataset = MADFD_dataset(dir_root=root, split='train')
    validate_dataset= MADFD_dataset(dir_root=root, split='val')
    # train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    # validate_sampler=torch.utils.data.distributed.DistributedSampler(validate_dataset)
    train_sampler = BatchSampler(train_dataset, batch_size=batch_size, drop_last=False)
    validate_sampler = BatchSampler(validate_dataset, batch_size=batch_size, drop_last=False) 
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,sampler=train_sampler,pin_memory=True,num_workers=config.workers)
    validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size,sampler=validate_sampler,pin_memory=True,num_workers=config.workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add_argument
    # parser.add_argument('--seed', type=int, default=4487, help='random seed for reproducibility')
    parser.add_argument('--config', type=str, default='', help='training config') # TODO: default
    args = parser.parse_args()
    train(args)