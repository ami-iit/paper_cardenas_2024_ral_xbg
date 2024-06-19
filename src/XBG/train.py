import os
import sys
import random
import yaml
import wandb
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import models
from utils import evaluate, save_checkpoint, load_checkpoint
sys.path.insert( 0, os.path.abspath("../../") )
from preprocessing.ergoCubDataset import ergoCubDataset

# Distributed GPU Training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# set device
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1203'
    init_process_group(backend='nccl', rank=rank, world_size=world_size) # nvidida collective communication library
                
def run_train(rank, world_size, run):
    ddp_setup(rank, world_size)
    device = rank 

    # Hyperparameters
    config = run.config
    hidden_size = config.hidden_size 
    fusion_kernel_size = config.fusion_kernel_size 
    drop_rate = config.drop_rate
    random_state = config.random_state 
    learning_rate = config.learning_rate 
    num_epochs = config.num_epochs 
    batch_size = config.batch_size 
    img_size = config.img_size 
    steps = config.steps 
    lookahead = config.lookahead 
    subsampling = config.subsampling 
    augmentations = config.augmentations 
    velocities = config.velocities 
    fts = config.fts 
    currents = config.currents 
    depth = config.depth 
    rgb = config.rgb 
    threshold = config.threshold 
    control_mode = config.control_mode 
    path = config.path 
    records = config.records 
    val_records = config.val_records 
    checkpoint_path = config.checkpoint_path 
    checkpoint_epoch = config.checkpoint_epoch 

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    transformations = v2.Compose([v2.ToTensor(),
                                v2.Resize(img_size, antialias=True), 
                                v2.CenterCrop(img_size)])
    if augmentations:
        augmentations = v2.RandomApply([
                            v2.RandomChoice([v2.RandomErasing(p=0.3, scale=(0.01, 0.02), value=1),
                                            v2.Compose([v2.RandomZoomOut(fill=1, side_range=(1.0, 1.05), p=0.3), v2.Resize(img_size, antialias=True)]),
                                            v2.RandomRotation((-4,4), fill=1),
                                            v2.ElasticTransform(alpha=50, sigma=5, fill=1),
                                            v2.RandomEqualize(p=0.5),
                                            v2.ColorJitter(brightness=.5, hue=.3),
                                            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                                            v2.RandomPosterize(4),
                                            v2.RandomAutocontrast(p=0.5)])
                                        ], 0.3)  
    else:
        augmentations = None

    train_dataset = ergoCubDataset(path, records, 
                        steps=steps, lookahead=lookahead, subsampling=subsampling, depth=depth, rgb=rgb, threshold=threshold, velocities=velocities, fts=fts, currents=currents, control_mode=control_mode, 
                        transform=transformations, augmentations=augmentations)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset))
    print(train_dataset.__len__(), 'Training Samples')

    config['scaler_mean'] = train_dataset.scaler.mean
    config['scaler_std'] = train_dataset.scaler.std
    config['input_names'] = train_dataset.input_names
    config['output_names'] = train_dataset.output_names
    INPUT_SIZE = len(config.input_names) 
    OUTPUT_SIZE = len(config.output_names) 
    # val_dataset = ergoCubDataset(path, val_records, 
    #                     steps=steps, lookahead=lookahead, subsampling=subsampling, depth=depth, rgb=rgb, threshold=threshold, velocities=velocities, fts=fts, currents=currents, control_mode=control_mode,
    #                     transform=transformations, scaler=train_dataset.scaler)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize network
    print('\n=== Building the Network ===')
        
    model = eval(f" models.{config.model}({INPUT_SIZE}, {hidden_size}, {OUTPUT_SIZE}, {fusion_kernel_size}, {drop_rate}).to({device})")
    model, starting_epoch = load_checkpoint(model, checkpoint_path, checkpoint_epoch)
    
    model = DDP(model, device_ids=[device])

    # Loss and optimizer
    loss_fcn = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    print('\n=== Starting Training ===')
    for epoch in range(starting_epoch, num_epochs):
        for batch_index, (rgb_imgs, depth_imgs, sensors, targets) in tqdm(enumerate(train_loader), total=int(train_dataset.__len__()/batch_size/world_size)):
            # Move rgb_imgs to Cuda if possible
            rgb_imgs = rgb_imgs.to(device=device)
            depth_imgs = depth_imgs.to(device=device)
            targets = targets.to(device=device).float()
            sensors = sensors.to(device=device).float()
            
            # print('forward')
            scores = model(sensors, rgb_imgs, depth_imgs)
            full_loss = loss_fcn(scores, targets) # Full loss matrix
            element_loss = torch.mean(full_loss, 0) # Mean loss for each joint
            loss = torch.mean(full_loss) # Mean loss
            if torch.isnan(loss):
                print(loss)
                print(element_loss)
                print(torch.mean(full_loss))
                exit()
            # print('backward')
            optimizer.zero_grad()
            loss.backward()

            # print('G.Descent')
            optimizer.step()

        if device == 0: # Validation only on one GPU
            print(f'epoch {epoch+1}/{num_epochs} ---> loss: {loss}')
            log = dict(zip(config.output_names, element_loss))
            log["loss"] = loss
            log["epoch"] = epoch
            wandb.log(log)    
            save_checkpoint(model, optimizer, loss, epoch, run, train_dataset.scaler, config)
        
    destroy_process_group()
                
def main():
    world_size = torch.cuda.device_count()
    run = wandb.init(project="test", job_type="train", group="DDP", tags=['v1', 'Sweep'])
    mp.spawn(run_train, args=(world_size, run), nprocs=world_size)
    wandb.finish()

if __name__ == "__main__":
    tune = False
    if tune:
        with open('tuning.yaml', 'r') as file:
            SWEEP_CONFIG = yaml.safe_load(file)
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=SWEEP_CONFIG['model'])
        wandb.agent(sweep_id, main, count=1)
    else:
        with open('training.yaml', 'r') as file:
            RUN_CONFIG = yaml.safe_load(file)
        print(RUN_CONFIG)    
        world_size = torch.cuda.device_count()
        run = wandb.init(project=RUN_CONFIG['model'], job_type="train", group="DDP", tags=[RUN_CONFIG['model'], 'test'], config=RUN_CONFIG)
        mp.spawn(run_train, args=(world_size, run), nprocs=world_size)
        wandb.finish()




