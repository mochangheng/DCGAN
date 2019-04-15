import numpy as np
import torch
from inception import InceptionV3
from scipy import linalg

device = torch.device('cuda:0')
latent_dim = 100

def get_batch_activations(batch, inception):
    inception.eval()
    # batch = batch.to(device)
    pred = inception(batch)[0]
    return pred.detach().cpu().numpy().reshape(batch.size()[0], -1)

def get_generated_activations(G, inception, batch_size=50, n_img=10000):
    n_batches = n_img // batch_size
    n_img = n_batches * batch_size
    pred_arr = np.empty((n_img, 2048))

    for i in range(batch_size):
        begin = i * batch_size
        end = begin + batch_size
        latents = torch.randn(batch_size, latent_dim).to(device)
        fake_batch = G(latents)
        pred_arr[begin:end] = get_batch_activations(fake_batch, inception)
        
    return pred_arr

def compute_stats(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
        

def get_fid(G, path):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx], normalize_input=False)
    inception = inception.to(device)

    generated_activations = get_generated_activations(G, inception, n_img=50000)
    mu_fake, sigma_fake = compute_stats(generated_activations)

    f = np.load(path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    return fid

def get_real_fid(batch):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx], normalize_input=False)
    inception = inception.to(device)

    real_activations = get_batch_activations(batch, inception)
    mu_fake, sigma_fake = compute_stats(real_activations)
    f = np.load('stats/fid_stats_lsun_train.npz')
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    print(np.max(mu_fake), np.max(mu_real), np.max(sigma_fake), np.max(sigma_real))
    fid = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    return fid

