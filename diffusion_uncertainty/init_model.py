from argparse import Namespace
from functools import singledispatch
import math
from typing import Optional, overload

import yaml
import torch
import numpy as np

from diffusion_uncertainty.guided_diffusion.unet_openai import EncoderUNetModel, UNetModel
from diffusion_uncertainty.paths import MODELS
from diffusion_uncertainty.uvit import load_pretrained_models
from diffusion_uncertainty.uvit.load_pretrained_models import load_autoencoder_uvit, load_uvit, load_uvit_scheduler
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.unets import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def init_guided_diffusion_imagenet128(load_checkpoint=True):
    params = dict(image_size=128, in_channels=3, model_channels=256, out_channels=6, num_res_blocks=2, attention_resolutions=(4,8,16), dropout=0.0, channel_mult=(1, 1, 2, 3, 4), num_classes=1000, use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=4, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=False)

    # load model and scheduler
    ddpm = UNetModel(**params)

    if load_checkpoint:
        ddpm.load_state_dict(torch.load(MODELS / f'128x128_diffusion.pt'))

    return ddpm


def init_guided_diffusion_imagenet128_and_scheduler(load_checkpoint=True):
    ddpm = init_guided_diffusion_imagenet128(load_checkpoint=load_checkpoint)

    scheduler = init_scheduler_imagenet128()
    return ddpm, scheduler

def init_scheduler_imagenet128():
    scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, beta_schedule='linear')
    return scheduler


def init_guided_diffusion_imagenet64(load_checkpoint=True, dropout: float = 0.1):

    hparams = dict(
        image_size=64, in_channels=3, model_channels=192, out_channels=6, num_res_blocks=3, attention_resolutions=(2, 4, 8), dropout=dropout, channel_mult=(1, 2, 3, 4), num_classes=1000, use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=64, num_heads_upsample=4, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=True
    )

    model = UNetModel(**hparams)

    if load_checkpoint:
        model.load_state_dict(state_dict=torch.load(MODELS / f"64x64_diffusion.pt"))

    model = model.to(torch.float32)

    return model

def create_classifier_openai_imagenet(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

def init_guided_diffusion_imagenet128_classifier():
    hyperparams = dict(
        image_size=128,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

    return create_classifier_openai_imagenet(**hyperparams)



def init_guided_diffusion_imagenet64_classifier() -> EncoderUNetModel:
    hyperparams = dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )

    return create_classifier_openai_imagenet(**hyperparams)

def init_guided_diffusion_imagenet64_and_scheduler(load_checkpoint=True, dropout=0.1):
    ddpm = init_guided_diffusion_imagenet64(load_checkpoint=load_checkpoint, dropout=dropout)

    scheduler = init_scheduler_imagenet64()
    return ddpm, scheduler

def init_scheduler_imagenet64():
    betas = betas_for_alpha_bar(1000, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    scheduler = DDPMScheduler(trained_betas=betas, num_train_timesteps=1000, beta_schedule='cosine')
    return scheduler


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def init_ddpm_cifar_10(dropout) -> UNet2DModel:
    model_id = "google/ddpm-cifar10-32"

    model: UNet2DModel = UNet2DModel.from_pretrained(model_id)

    if dropout is not None:
        print(f"Setting dropout of ddpm cifar10 to {dropout}")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
    else:
        print("Using default dropout for ddpm cifar10")

    return model

def init_ddpm_cifar_10_scheduler() -> DDPMScheduler:
    model_id = "google/ddpm-cifar10-32"
    scheduler = DDPMScheduler.from_pretrained(model_id)

    return scheduler

def init_ddpm_cifar_10_and_scheduler(dropout: Optional[float] = None) -> tuple[UNet2DModel, DDPMScheduler]:
    """
    Initializes the DDPM CIFAR-10 model and scheduler.

    Returns:
        A tuple containing the initialized model and scheduler.
    """
    model = init_ddpm_cifar_10(dropout)
    scheduler = init_ddpm_cifar_10_scheduler()



    return model, scheduler


def init_uvit_imagenet_256(device='cuda') -> UViTAE:
    uvit = load_pretrained_models.load_uvit(256, device)
    ae = load_pretrained_models.load_autoencoder_uvit(device)
    uvitae = UViTAE(uvit, ae)
    return uvitae

def init_uvit_imagenet_256_with_scheduler(device='cuda') -> tuple[UViTAE, DDPMScheduler]:
    uvitae = init_uvit_imagenet_256(device=device)
    scheduler = load_pretrained_models.load_uvit_scheduler()
    return uvitae, scheduler

def init_uvit_imagenet_512() -> UViTAE:
    uvit = load_pretrained_models.load_uvit(512)
    ae = load_pretrained_models.load_autoencoder_uvit()
    uvitae = UViTAE(uvit, ae)

    return uvitae

def init_uvit_imagenet_512_with_scheduler() -> tuple[UViTAE, DDPMScheduler]:
    uvitae = init_uvit_imagenet_512()
    scheduler = load_pretrained_models.load_uvit_scheduler()

    return uvitae, scheduler


@singledispatch
def instantiate_model_scheduler(args):
    """
    Instantiate the model scheduler.

    Args:
        args: The arguments for the model scheduler. Can be Namespace or string (dataset name)
    """
    raise NotImplementedError

@instantiate_model_scheduler.register
def _instantiate_model_scheduler(dataset_name: str):
    if  dataset_name == 'imagenet64':
        model, scheduler = init_guided_diffusion_imagenet64_and_scheduler()
    elif dataset_name == 'imagenet128':
        model, scheduler = init_guided_diffusion_imagenet128_and_scheduler()
    elif dataset_name == 'imagenet256':
        uvit = load_uvit(256, 'cuda')
        autoencoder = load_autoencoder_uvit('cuda')
        model = UViTAE(uvit, autoencoder)
        scheduler = load_uvit_scheduler()

    elif dataset_name == 'imagenet512':
        uvit = load_uvit(512, 'cuda')
        autoencoder = load_autoencoder_uvit('cuda')
        model = UViTAE(uvit, autoencoder)
        scheduler = load_uvit_scheduler()
    elif dataset_name == 'cifar10':
        model, scheduler = init_ddpm_cifar_10_and_scheduler(0.0)
    else:
        raise ValueError(f'invalid dataset name {dataset_name}')
    return model,scheduler


@instantiate_model_scheduler.register
def _instantiate_model_scheduler(args: Namespace):
    first_dataset_folder = args.dataset_folders[0]
    with open(first_dataset_folder / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_name: str = config['dataset']
    return instantiate_model_scheduler(dataset_name)

init_model_scheduler = instantiate_model_scheduler

@singledispatch
def init_scheduler(args) -> SchedulerMixin:
    """
    Initialize the scheduler.

    Args:
        args: The arguments for the scheduler. Can be Namespace or string (dataset name)
    """
    raise NotImplementedError


@init_scheduler.register
def _init_scheduler(dataset_name: str):
    if dataset_name == 'imagenet64':
        scheduler = init_scheduler_imagenet64()
    elif dataset_name == 'imagenet128':
        scheduler = init_scheduler_imagenet128()
    elif dataset_name == 'imagenet256':
        scheduler = load_pretrained_models.load_uvit_scheduler()
    elif dataset_name == 'imagenet512':
        scheduler = load_pretrained_models.load_uvit_scheduler()
    elif dataset_name == 'cifar10':
        scheduler = init_ddpm_cifar_10_scheduler()
    else:
        raise ValueError(f'invalid dataset name {dataset_name}')
    return scheduler

@init_scheduler.register
def _init_scheduler(args: Namespace):
    first_dataset_folder = args.dataset_folders[0]
    with open(first_dataset_folder / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_name: str = config['dataset']
    return init_scheduler(dataset_name)


