import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class G_basic(nn.Module):
    def __init__(self, res, label_size=0):
        super(G_basic, self).__init__()
        max_scale = int(np.log2(res))
        assert res == 2**max_scale

        if label_size>0:
            self.label_size = label_size
            self.label_in = nn.Linear(label_size, 50)
            self.latent_in = nn.Linear(100, 50)

        current_channel = 1024
        self.deconv1 = nn.ConvTranspose2d(100, current_channel, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(current_channel)
        self.relu1 = nn.ReLU()

        self.layers = nn.ModuleList([])
        for scale in range(3, max_scale):
            self.layers.append(nn.ConvTranspose2d(current_channel, int(current_channel/2), kernel_size=4, stride=2, padding=1))
            current_channel = int(current_channel/2)
            self.layers.append(nn.BatchNorm2d(current_channel))
            self.layers.append(nn.ReLU())
        self.last_deconv = nn.ConvTranspose2d(current_channel, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, latent, label=None):
        if label:
            assert self.label_size > 0
            label = self.label_in(label)
            latent = self.latent_in(latent)
            latent = torch.cat([label, latent], dim=1)

        latent = latent.view(latent.shape[0],latent.shape[1],1,1)
        x = self.deconv1(latent)
        x = self.bn1(x)
        x = self.relu1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.tanh(self.last_deconv(x))
        return x

class D_basic(nn.Module):
    def __init__(self, res, label_size=0):
        super(D_basic, self).__init__()
        max_scale = int(np.log2(res))
        assert res == 2**max_scale

        current_channel = int(1024/(2**(max_scale-3)))
        self.conv1 = nn.Conv2d(3, current_channel, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.layers = nn.ModuleList([])

        for scale in range(max_scale-1, 2, -1):
            self.layers.append(nn.Conv2d(current_channel, current_channel*2, kernel_size=4, stride=2, padding=1))
            current_channel *= 2
            self.layers.append(nn.BatchNorm2d(current_channel))
            self.layers.append(nn.LeakyReLU(0.2))
        
        self.last_conv = nn.Conv2d(current_channel, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, image):
        x = self.conv1(image)
        x = self.lrelu1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_conv(x)
        return torch.squeeze(x)
