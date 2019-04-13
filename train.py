import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN
from torchvision import transforms
from tensorboardX import SummaryWriter

from network import D_basic, G_basic
from loss import D_logistic, G_logistic_nonsaturating
from util import *
from evaluate import *
from training import *
from metric import get_fid

num_iter = 1000000
run_name = 'LSUN_bedroom'
batch_size = 128
image_size = 64

cur_iter = 0
iter_per_tick = 1000
ticks_per_snapshot = 10
result_dir = 'results/'
device = torch.device("cuda:0")

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

run_dir = create_new_run(result_dir, run_name)
logger = Logger(os.path.join(run_dir, 'log.txt'))
writer = SummaryWriter(run_dir)

LSUN_dataset = LSUN(root='../lsun/', classes=['bedroom_train'], transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

dataloader = DataLoader(LSUN_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

D = D_basic(image_size)
G = G_basic(image_size)

D = D.to(device)
G = G.to(device)

D.apply(weights_init)
G.apply(weights_init)

D_optim = Adam(D.parameters(), lr=1e-4, weight_decay=5e-4)
G_optim = Adam(G.parameters(), lr=1e-4, weight_decay=5e-4)

iter_counter = 0
tick_counter = 0
running_D_loss = 0
running_G_loss = 0

# Setting up grid
grid_latents = get_grid_latents(6,6).to(device)

logger.log('Starting training..')

while cur_iter < num_iter:

    for real_batch,_ in dataloader:

        D.train()
        G.train()

        cur_iter += 1
        iter_counter += 1

        # Train discriminator
        D_optim.zero_grad()

        real_batch = real_batch.to(device)
        real_out = D(real_batch)

        latent = torch.randn(batch_size, 100)
        latent = latent.to(device)
        fake_batch = G(latent)
        fake_out = D(fake_batch.detach())

        D_loss = D_logistic(real_out, fake_out)
        D_loss.backward()
        D_optim.step()
        running_D_loss += D_loss.item()

        # Train generator
        G_optim.zero_grad()

        fake_out = D(fake_batch)
        G_loss = G_logistic_nonsaturating(fake_out)
        G_loss.backward()
        G_optim.step()
        running_G_loss += G_loss.item()

        if iter_counter >= iter_per_tick:
            # print(running_D_loss, iter_counter)
            logger.log('[Iteration {:07d}] D_loss: {:.5f} G_loss: {:.5f}'.format(cur_iter, running_D_loss/iter_counter, running_G_loss/iter_counter))
            # Log to tensorboard
            writer.add_scalar('loss/D_loss', running_D_loss/iter_counter, cur_iter)
            writer.add_scalar('loss/G_loss', running_G_loss/iter_counter, cur_iter)

            # Save grid
            G.eval()
            save_grid_images(G, grid_latents, 6, 6, run_dir, cur_iter)

            iter_counter = 0
            tick_counter += 1
            running_D_loss = 0
            running_G_loss = 0
        
        if tick_counter >= ticks_per_snapshot:
            tick_counter = 0

            # Evaluate
            fid = get_fid(G, 'stats/fid_stats_lsun_train.npz')
            logger.log('FID: {}'.format(fid))
            writer.add_scalar('loss/FID', fid, cur_iter)

            # Save model
            save_dict = {
                'G': G.state_dict(),
                'D': D.state_dict(),
            }
            torch.save(save_dict, os.path.join(run_dir, 'network-snapshot-{:07d}.pkl'.format(cur_iter)))
            if cur_iter >= num_iter:
                break
        
logger.log('Finished training.')
logger.close()
