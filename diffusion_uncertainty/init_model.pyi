from argparse import Namespace
from typing import Optional, Tuple, Union, overload

import torch
from diffusion_uncertainty.guided_diffusion.unet_openai import EncoderUNetModel, UNetModel
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.unets import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

def init_guided_diffusion_imagenet128(load_checkpoint: bool = True) -> UNetModel: ...

def init_guided_diffusion_imagenet128_and_scheduler(load_checkpoint: bool = True) -> Tuple[UNetModel, DDPMScheduler]: ...

def init_scheduler_imagenet128() -> DDPMScheduler: ...

def init_guided_diffusion_imagenet64(load_checkpoint: bool = True, dropout: float = 0.1) -> UNetModel: ...

def init_guided_diffusion_imagenet64_and_scheduler(load_checkpoint: bool = True, dropout: float = 0.1) -> Tuple[UNetModel, DDPMScheduler]: ...

def init_scheduler_imagenet64() -> DDPMScheduler: ...

def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar: Callable[[float], float], max_beta: float = 0.999) -> np.ndarray: ...

def init_ddpm_cifar_10(dropout: Optional[float]) -> UNet2DModel: ...

def init_ddpm_cifar_10_scheduler() -> DDPMScheduler: ...

def init_ddpm_cifar_10_and_scheduler(dropout: Optional[float] = None) -> Tuple[UNet2DModel, DDPMScheduler]: ...

def init_uvit_imagenet_256(device: str = 'cuda') -> UViTAE: ...

def init_uvit_imagenet_256_with_scheduler(device: str = 'cuda') -> Tuple[UViTAE, DDPMScheduler]: ...

def init_uvit_imagenet_512() -> UViTAE: ...

def init_uvit_imagenet_512_with_scheduler() -> Tuple[UViTAE, DDPMScheduler]: ...

@overload
def instantiate_model_scheduler(args: Namespace) -> Tuple[Union[UNetModel, UNet2DModel, UViTAE], SchedulerMixin]: ...

@overload
def instantiate_model_scheduler(args: str) -> Tuple[Union[UNetModel, UNet2DModel, UViTAE], SchedulerMixin]: ...

@overload
def init_model_scheduler(args: Namespace) -> Tuple[Union[UNetModel, UNet2DModel, UViTAE], SchedulerMixin]: ...

@overload
def init_model_scheduler(args: str) -> Tuple[Union[UNetModel, UNet2DModel, UViTAE], SchedulerMixin]: ...

@overload
def init_scheduler(args: Namespace) -> SchedulerMixin: ...

@overload
def init_scheduler(args: str) -> SchedulerMixin: ...

def init_guided_diffusion_imagenet64_classifier() -> EncoderUNetModel: ...
def init_guided_diffusion_imagenet128_classifier() -> EncoderUNetModel: ...