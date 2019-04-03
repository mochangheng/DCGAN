import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import LSUN
from torchvision import transforms
from network import D_basic, G_basic
from loss import D_logistic, G_logistic_nonsaturating

num_iter = 100000
cur_iter = 0
D_iter_per_G = 1
iter_per_tick = 100
ticks_per_snapshot = 10
batch_size = 32
image_size = 64

LSUN_dataset = LSUN(root='./datasets/LSUN', classes=['bedroom_train'], transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))
dataloader = DataLoader(LSUN_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0")

D = D_basic(256)
G = G_basic(256)

D = D.to(device)
G = G.to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

D.apply(weights_init)
G.apply(weights_init)

D_optim = Adam(D.parameters(), lr=1e-4)
G_optim = Adam(G.parameters(), lr=1e-4)

iter_counter = 0
tick_counter = 1

while cur_iter < num_iter:
    # Train phase
    D.train()
    G.train()

    running_D_loss = 0
    running_G_loss = 0

    for i, real_batch in enumerate(dataloader, 0):

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
            print('[Tick {:06d}] D_loss: {:.6f} G_loss: {:.6f}'.format(cur_iter, running_D_loss/iter_counter, running_G_loss/iter_counter))
            iter_counter = 0
            tick_counter += 1
        
        if tick_counter >= ticks_per_snapshot:
            tick_counter = 0
    
    