from sched import scheduler
import sys
from path import Path

from diffusion_uncertainty.uvit.load_pretrained_models import load_uvit
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import torch
from diffusion_uncertainty.uvit.load_pretrained_models import load_uvit, load_autoencoder_uvit
from diffusion_uncertainty.uvit.uvit import UViT
from diffusion_uncertainty.paths import MODELS, UVIT_IMAGENET64_CKPT
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import save_image



def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

@torch.no_grad()
def test_generate_imagenet_uvit():

    # generate images
    imagenet_uvit = load_uvit(256, 'cpu')

    autoencoder = load_autoencoder_uvit('cpu')


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


    ddim_scheduler = DDIMScheduler.from_config(scheduler_config)
    ddim_scheduler.set_timesteps(20)
    
    image = torch.randn(4, 4, 32, 32)

    y = torch.randint(0, 1000, (4,))

    for t in ddim_scheduler.timesteps:
        t_tensor = torch.zeros(4, dtype=torch.long).fill_(t)
        eps = imagenet_uvit(image, t_tensor, y)
        image = ddim_scheduler.step(eps, t, image, return_dict=True).prev_sample

    image = autoencoder.decode(image)

    assert image.shape == (4, 3, 32*8, 32*8)

    image = make_grid(image, nrow=2, normalize=True)

    save_image(image, 'tests/uvit_gen.png')