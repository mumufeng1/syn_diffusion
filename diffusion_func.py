from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
from IPython.display import display
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm_notebook
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU
from torch.optim import Adam
from tqdm import tqdm_notebook
from torchvision.utils import save_image
import matplotlib
import math
from inspect import isfunction
from functools import partial
import scipy
from scipy.special import rel_entr
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.image as mpimg
import glob
from PIL import Image
from func import *


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,  noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # print  (sqrt_alphas_cumprod_t , sqrt_one_minus_alphas_cumprod_t , t)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):

    if noise is None:
        noise = torch.randn_like(x_start)  #  guass noise
    x_noisy = q_sample(x_start=x_start, t=t,
                       sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                       sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                       noise=noise)  # this is the auto generated noise given t and Noise
    predicted_noise = denoise_model(x_noisy, t)  # this is the predicted noise given the model and step

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

# FUNCTION CHANGED ADD CONDITIONING
#
@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    # print (x.shape, 'x_shape')
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, time=t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# FUNCTION CHANGED ADD CONDITIONING
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i,
                       betas=betas,
                       sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                       sqrt_recip_alphas=sqrt_recip_alphas,
                       posterior_variance=posterior_variance)
        imgs.append(img.cpu().numpy())
    return imgs


# FUNCTION CHANGED ADD CONDITIONING


@torch.no_grad()
def sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, 4, image_size),
                         timesteps=timesteps, betas=betas,
                         sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                         sqrt_recip_alphas=sqrt_recip_alphas,
                         posterior_variance=posterior_variance)
