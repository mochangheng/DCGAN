import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN
from torchvision import transforms

from training import *

train_config = {
    'num_iter': 1000000,
    'batch_size': 128,
    'image_size': 64,
    'iter_per_tick': 1000,
    'ticks_per_snapshot': 10,
    'device': torch.device("cuda:0"),
}

run_name = 'LSUN_bedroom'
lsun_dir = '../lsun/'
result_dir = 'results/'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

dataset = LSUN(root=lsun_dir, classes=['bedroom_train'], transform=transforms.Compose([
    transforms.Resize(train_config['image_size']),
    transforms.CenterCrop(train_config['image_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

train_config['dataloader'] = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0)

train_session = TrainSession(run_name, result_dir)

train_session.training_loop(**train_config)
        
