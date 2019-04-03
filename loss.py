import numpy as np
import torch

def G_logistic_nonsaturating(fake_out):
    loss = -torch.mean(torch.log(torch.sigmoid(fake_out)))
    return loss

def D_logistic(real_out, fake_out):
    real_loss = -torch.mean(torch.log(torch.sigmoid(real_out)))
    fake_loss = -torch.mean(torch.log(1-torch.sigmoid(fake_out)))
    return real_loss+fake_loss