import numpy as np
import torch

_EPSILON = 1e-10
device = torch.device("cuda:0")

def G_logistic_nonsaturating(fake_out):
    eps = torch.Tensor([_EPSILON]).to(device)
    fake_prob = torch.max(torch.sigmoid(fake_out), eps)
    loss = -torch.mean(torch.log(fake_prob))
    return loss

def D_logistic(real_out, fake_out):
    eps = torch.Tensor([_EPSILON]).to(device)
    real_prob = torch.max(torch.sigmoid(real_out), eps)
    fake_prob = torch.max(1-torch.sigmoid(fake_out), eps)
    real_loss = -torch.mean(torch.log(real_prob))
    fake_loss = -torch.mean(torch.log(fake_prob))
    return real_loss+fake_loss

def G_energy(fake_out):
    pass