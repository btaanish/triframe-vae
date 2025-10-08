import torch
import numpy as np
from scipy import linalg
import torch.nn.functional as F
import os

codes1 = '/content/drive/My Drive/Colab Notebooks/Soft-Fusion-FYP/a45.pth' # ground truth
print(os.path.exists(codes1))
# codes1 = '/root/autodl-tmp/SDFusion/test_ground_truth/codes.pth'
# codes2 = '/root/autodl-tmp/SDFusion/data/results_nopre_large_dset/codes.pth'  # 21.99817
# codes2 = '/root/autodl-tmp/SDFusion/data/results_nopre_small_dset/codes.pth'  # 24.15443
# codes2 = '/root/autodl-tmp/SDFusion/data/results_pretrained_large_dset/codes.pth'  # 20.2295
# codes2 = '/root/autodl-tmp/SDFusion/data/results_pretrained_small_dset/codes.pth'  # 23.67713
codes2 = '/content/drive/My Drive/Colab Notebooks/Soft-Fusion-FYP/fold based extension 1dof soft actuator 6 folds short Model 3.pth'

def average_pooling(tensor):
    return F.avg_pool3d(tensor, kernel_size=2, stride=2)

codes1 = torch.load(codes1)
codes2 = torch.load(codes2)

print(f"codes1: {codes1.shape}, codes2: {codes2.shape}")

# supress to 3, 64
# codes1 = torch.mean(codes1, (2, 4)).reshape(-1, 3*16)
# codes2 = torch.mean(codes2, (2, 4)).reshape(-1, 3*16)
# codes1 = codes1.reshape(-1, 3*16**3)
# codes2 = codes2.reshape(-1, 3*16**3)

# [Bs, 3, 16, 16, 16] -> [Bs, 3, 8, 8, 8], spatial average pooling
codes1 = average_pooling(codes1).reshape(-1, 3*8**3)
codes2 = average_pooling(codes2).reshape(-1, 3*8**3)

print(f"codes1: {codes1.shape}, codes2: {codes2.shape}")

# calculate Fréchet Distance
mu1 = torch.mean(codes1, 0)
mu2 = torch.mean(codes2, 0)

sigma1 = torch.cov(codes1.T)
sigma2 = torch.cov(codes2.T)

print(f"mu1: {mu1.shape}, mu2: {mu2.shape}")
print(f"sigma1: {sigma1.shape}, sigma2: {sigma2.shape}")

# FD = torch.norm(mu1 - mu2) + torch.trace(sigma1 + sigma2 - 2 * torch.sqrt(sigma1 @ sigma2))
# print(FD)

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

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


fd = calculate_frechet_distance(mu1.cpu().numpy(), sigma1.cpu().numpy(), mu2.cpu().numpy(), sigma2.cpu().numpy())
print(fd)