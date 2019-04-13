import numpy as np
from PIL import Image
import torch
import os

def get_grid_latents(h, w):
    return torch.randn(h*w, 100)

def save_grid_images(G, grid_latents, h, w, run_dir, cur_iter):
    fake_images = G(grid_latents).detach().cpu().numpy()
    fake_images = (fake_images*0.5)+0.5
    fake_images = fake_images.transpose(0,2,3,1)

    rows = []
    for i in range(h):
        cols = []
        for j in range(w):
            cols.append(fake_images[i*h+j,:,:,:])
        cols = np.concatenate(cols, axis=1)
        rows.append(cols)
    fake_grid = np.concatenate(rows, axis=0)
    fake_grid = Image.fromarray(np.uint8(fake_grid*255))
    fake_grid.save(os.path.join(run_dir, 'fakes-{:06d}.png'.format(cur_iter)))
