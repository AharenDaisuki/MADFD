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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from models.MAT import MAT
# from datasets.dataset import DeepfakeDataset
from datasets.custom_dataset import MADFD_dataset
from AGDA import AGDA
import cv2
from utils import dist_average,ACC
assert torch.cuda.is_available()

PERIOD = 100

def train(args):
    config = args.config
    
    seed = config.seed
    root = config.root
    batch_size = config.batch_size
    lr = config.learning_rate
    betas = config.adam_betas
    weight_decay = config.weight_decay
    step_size = config.scheduler_step
    gamma = config.scheduler_gamma
    epoch_n = config.epochs
    resume = config.ckpt
    
    start_epoch = 0
    logs = {}

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
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size,sampler=train_sampler,pin_memory=True,num_workers=config.workers)
    # validate_loader = DataLoader(validate_dataset, batch_size=config.batch_size,sampler=validate_sampler,pin_memory=True,num_workers=config.workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=train_sampler)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size,sampler=validate_sampler)

    model = MAT(**config.net_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if resume:
        pass

    torch.cuda.empty_cache()
    for epoch in range(start_epoch, epoch_n):
        logs['epoch'] = epoch
        # train_sampler.set_epoch(epoch)
        # train_sampler.dataset.next_epoch()
        # run(logs=logs,data_loader=train_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,AG=AG,phase='train')
        # run(logs=logs,data_loader=validate_loader,net=net,optimizer=optimizer,local_rank=local_rank,config=config,phase='valid')
        # net.module.auxiliary_loss.alpha*=config.alpha_decay
        scheduler.step()
        if epoch % PERIOD == 0:
            torch.save(
                {
                    'logs': logs,
                    'state_dict': model.module.state_dict(),
                    'optimzer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                },
                f'' # TODO: checkpoint file path
            )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add_argument
    # parser.add_argument('--seed', type=int, default=4487, help='random seed for reproducibility')
    parser.add_argument('--config', type=str, default='', help='training config') # TODO: default
    args = parser.parse_args()
    train(args)