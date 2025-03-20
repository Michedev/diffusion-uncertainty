from path import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from diffusion_uncertainty.paths import MODELS
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_uncertainty.guided_diffusion.unet_openai import UNetModel
from torchvision.utils import save_image


@torch.no_grad()
def test_generation_imagenet128():
    params = dict(image_size=128, in_channels=3, model_channels=256, out_channels=6, num_res_blocks=2, attention_resolutions=(4,8,16), dropout=0.0, channel_mult=(1, 1, 2, 3, 4), num_classes=1000, use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=4, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=False)

    # load model and scheduler
    ddpm = UNetModel(**params)
    if not torch.cuda.is_available():
        ddpm.load_state_dict(torch.load(MODELS / f'128x128_diffusion.pt', map_location='cpu'))
    else:
        ddpm.load_state_dict(torch.load(MODELS / f'128x128_diffusion.pt'))

    scheduler = DDIMScheduler(

    )

    scheduler.set_timesteps(20)

    image = torch.randn(4, 3, 128, 128)
    y = torch.randint(0, 1000, (4,))

    for t in tqdm(scheduler.timesteps):
        t_tensor = torch.zeros(4, dtype=torch.long).fill_(t)    

        eps = ddpm(image, t_tensor, y=y)[:,:3]
        
        image = scheduler.step(eps, timestep=t, sample=image).prev_sample

    assert image is not None

    image = image * 0.5 + 0.5


    save_image(image, Path(__file__).parent / 'test.png')



@torch.no_grad()
def test_generation_imagenet128_without_class_should_not_work():
    params = dict(image_size=128, in_channels=3, model_channels=256, out_channels=6, num_res_blocks=2, attention_resolutions=(4,8,16), dropout=0.0, channel_mult=(1, 1, 2, 3, 4), num_classes=1000, use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=4, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=False)

    # load model and scheduler
    ddpm = UNetModel(**params)

    ddpm.load_state_dict(torch.load(MODELS / f'128x128_diffusion.pt'))

    scheduler = DDIMScheduler(

    )

    scheduler.set_timesteps(20)

    image = torch.randn(4, 3, 128, 128)
    # y = torch.randint(0, 1000, (4,))

    for t in tqdm(scheduler.timesteps):
        t_tensor = torch.zeros(4, dtype=torch.long).fill_(t)    
        try:
            eps = ddpm(image, t_tensor,)[:,:3]
        except RuntimeError:
            return
        except AssertionError:
            return
        assert False, "Should not be reached"
        image = scheduler.step(eps, timestep=t, sample=image).prev_sample

    assert image is not None

    image = image * 0.5 + 0.5


    save_image(image, Path(__file__).parent / 'imagent_128_without_class.png')