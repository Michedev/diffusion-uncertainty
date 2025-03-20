from typing import Literal
import torch
from diffusion_uncertainty.paths import MODELS
from diffusion_uncertainty.uvit.uvit import UViT
from beartype import beartype
from diffusion_uncertainty.uvit.autoencoder import get_model as _get_autoencoder_model
import os
import subprocess

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

@beartype
def load_uvit(image_size: Literal[256, 512], device: str = 'cuda'):
    """
    Loads a pretrained UViT model for image classification. Downloads the model if it does not exist.

    Args:
        image_size (int): The size of the input image. Must be either 256 or 512.
        device (str, optional): The device to load the model on. Defaults to 'cuda'.

    Returns:
        nnet (UViT): The loaded UViT model.
    """
    path_to_ckpt = MODELS / f'imagenet{image_size}_uvit_huge.pth'
    if image_size == 256 and not path_to_ckpt.exists():
        print('Downloading UViT model for image size 256...')
        subprocess.run(['gdown', '13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u', '-O', str(path_to_ckpt.absolute())])
        print('UViT model for image size 256 downloaded in the path:', path_to_ckpt.absolute())
    elif image_size == 512 and not path_to_ckpt.exists():
        print('Downloading UViT model for image size 512...')
        subprocess.run(['gdown', '1uegr2o7cuKXtf2akWGAN2Vnlrtw5YKQq', '-O', str(path_to_ckpt.absolute())])
        print('UViT model for image size 512 downloaded in the path:', path_to_ckpt.absolute())

    patch_size = 2 if image_size == 256 else 4
    z_size = image_size // 8 
    nnet = UViT(img_size=z_size, patch_size=patch_size, in_chans=4, embed_dim=1152, depth=28, num_heads=16, num_classes=1001, conv=False) 
    nnet.to(device) 
    nnet.load_state_dict(torch.load(path_to_ckpt, map_location=device)) 
    nnet.eval()

    return nnet


def load_uvit_scheduler() -> DDPMScheduler:
    scheduler_config = {
    "_class_name": "PNDMScheduler",
    "_diffusers_version": "0.6.0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
    "clip_sample": False
    }


    ddpm_scheduler = DDPMScheduler.from_config(scheduler_config)
    assert isinstance(ddpm_scheduler, DDPMScheduler), f"Expected DDPMScheduler, got {type(ddpm_scheduler)}"
    return ddpm_scheduler



def load_autoencoder_uvit(device: str = 'cuda'):
    """
    Loads the pretrained autoencoder model. Downloads the model if it does not exist.

    Args:
        device (str): The device to load the model on. Defaults to 'cuda'.

    Returns:
        autoencoder: The loaded autoencoder model.
    """
    ckpt_path = MODELS / 'autoencoder_kl_ema.pth'
    if not ckpt_path.exists():
        print('Downloading autoencoder model...')
        subprocess.run(['gdown',  '10nbEiFd4YCHlzfTkJjZf45YcSMCN34m6', '-O', str(ckpt_path.absolute())], shell=True, check=True)
        print('Autoencoder model downloaded in the path:', ckpt_path.absolute())
    autoencoder = _get_autoencoder_model(ckpt_path)
    autoencoder.eval()
    autoencoder.to(device)
    return autoencoder

@torch.no_grad()
def sample_uvit(uvit: UViT, scheduler, autoencoder: torch.nn.Module, y: torch.Tensor, x_T: torch.Tensor | None = None):
    """
    Sample UViT model with given inputs.

    Args:
        uvit (UViT): The UViT model.
        scheduler: The scheduler object.
        autoencoder (torch.nn.Module): The autoencoder model.
        y: The input tensor y.
        x_T: The input tensor x_T.

    Returns:
        torch.Tensor: The output image tensor.
    """
    if x_T is None:
        x_T = torch.randn(y.shape[0], 4, uvit.img_size, uvit.img_size)
    assert x_T.shape[0] == y.shape[0], f"Batch size of x_T and y must be the same, got {x_T.shape[0]=} and {y.shape[0]=} respectively"
    image = x_T.clone()
    for t in scheduler.timesteps:
        t_tensor = torch.zeros(y.shape[0], dtype=torch.long).fill_(t)
        eps = uvit(image, t_tensor, y)
        image = scheduler.step(eps, t, image, return_dict=True).prev_sample

    image = autoencoder.decode(image)
    return image