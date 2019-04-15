import torch
from torch.optim import Adam
from network import *
from loss import *
from evaluate import *
from util import *
from metric import get_fid, get_real_fid
from tensorboardX import SummaryWriter

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_networks_by_type(network_type, image_size):
    if network_type == 'basic':
        D = D_basic(image_size)
        G = G_basic(image_size)

    return D, G

class TrainSession():
    def __init__(self, run_name, result_dir):
        self.run_dir = create_new_run(result_dir, run_name)
        self.logger = Logger(os.path.join(self.run_dir, 'log.txt'))
        self.writer = SummaryWriter(self.run_dir)

    def log(self, s):
        self.logger.log(s)

    def close(self):
        self.log('Finished training.')
        self.logger.close()

    def training_loop(self,
        dataloader,
        num_iter,
        batch_size,
        image_size,
        network_type = 'basic',
        iter_per_tick = 1000,
        ticks_per_snapshot = 10,
        D_lr = 1e-4,
        G_lr = 1e-4,
        device = torch.device('cuda:0')):

        iter_per_tick = 10
        ticks_per_snapshot = 1
        batch_size = 256

        self.log('Starting training..')

        D, G = get_networks_by_type(network_type, image_size)

        D = D.to(device)
        G = G.to(device)

        D.apply(weights_init)
        G.apply(weights_init)

        D_optim = Adam(D.parameters(), lr=D_lr, weight_decay=5e-4)
        G_optim = Adam(G.parameters(), lr=G_lr, weight_decay=5e-4)

        D_criterion = D_logistic
        G_criterion = G_logistic_nonsaturating

        iter_counter = 0
        tick_counter = 0
        running_D_loss = 0
        running_G_loss = 0

        # Setting up grid
        grid_latents = get_grid_latents(6,6).to(device)
        cur_iter = 0

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

                D_loss = D_criterion(real_out, fake_out)
                D_loss.backward()
                D_optim.step()
                running_D_loss += D_loss.item()

                # Train generator
                G_optim.zero_grad()

                fake_out = D(fake_batch)
                G_loss = G_criterion(fake_out)
                G_loss.backward()
                G_optim.step()
                running_G_loss += G_loss.item()

                if iter_counter >= iter_per_tick:
                    # print(running_D_loss, iter_counter)
                    self.log('[Iteration {:07d}] D_loss: {:.5f} G_loss: {:.5f}'.format(cur_iter, running_D_loss/iter_counter, running_G_loss/iter_counter))
                    # Log to tensorboard
                    self.writer.add_scalar('loss/D_loss', running_D_loss/iter_counter, cur_iter)
                    self.writer.add_scalar('loss/G_loss', running_G_loss/iter_counter, cur_iter)

                    # Save grid
                    G.eval()
                    save_grid_images(G, grid_latents, 6, 6, self.run_dir, cur_iter)

                    iter_counter = 0
                    tick_counter += 1
                    running_D_loss = 0
                    running_G_loss = 0
                
                if tick_counter >= ticks_per_snapshot:
                    tick_counter = 0

                    # Evaluate
                    fid = get_fid(G, 'stats/fid_stats_lsun_train.npz')
                    self.log('FID: {}'.format(fid))
                    self.writer.add_scalar('loss/FID', fid, cur_iter)

                    real_fid = get_real_fid(real_batch)
                    print(real_batch.size())
                    print(real_fid)

                    # Save model
                    save_dict = {
                        'G': G.state_dict(),
                        'D': D.state_dict(),
                    }
                    torch.save(save_dict, os.path.join(self.run_dir, 'network-snapshot-{:07d}.pkl'.format(cur_iter)))
                    if cur_iter >= num_iter:
                        return
        
        self.close()
        return