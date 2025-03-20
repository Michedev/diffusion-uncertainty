import sys
from path import Path
sys.path.insert(0, Path(__file__).parent.parent.absolute())

from argparse import Namespace
from tqdm import tqdm
import json
from diffusion_uncertainty.paths import DEBUG, IMAGENET128, FID
from diffusion_uncertainty.fid import load_real_fid_model
from diffusion_uncertainty.guided_diffusion.unet_openai import UNetModel, args_to_dict, create_model
from diffusion_uncertainty.init_model import init_guided_diffusion_imagenet128
import torch
import torchvision
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Compute FID score for real images')
    parser.add_argument('--num-samples', type=int, default=50000, help='Number of samples to use for computing FID score', dest='num_samples')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for computing FID score', dest='batch_size')
    parser.add_argument('--on-cpu', action='store_true', help='Use CPU for computing FID score', dest='on_cpu')
    parser.add_argument('--num-timesteps', type=int, default=50, help='Number of timesteps for diffusion', dest='num_timesteps')


    args = parser.parse_args()

    args = Namespace(clip_denoised=True, num_samples=args.num_samples, batch_size=args.batch_size, use_ddim=True, model_path='models/128x128_diffusion.pt', classifier_path='models/128x128_classifier.pt', classifier_scale=0.0, image_size=128, num_channels=256, num_res_blocks=2, num_heads=4, num_heads_upsample=-1, num_head_channels=-1, attention_resolutions='32,16,8', channel_mult='', dropout=0.0, class_cond=False, use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True, use_fp16=False, use_new_attention_order=False, learn_sigma=True, diffusion_steps=1000, noise_schedule='linear', timestep_respacing='ddim50', use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False, classifier_use_fp16=True, classifier_width=128, classifier_depth=2, classifier_attention_resolutions='32,16,8', classifier_use_scale_shift_norm=True, classifier_resblock_updown=True, classifier_pool='attention', on_cpu=args.on_cpu, num_timesteps=args.num_timesteps)

    if args.on_cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('WARNING: Using CPU for computing FID score. Consider using GPU for faster computation.')
    device = torch.device(device)
    print('Using device:', device)

    model = init_guided_diffusion_imagenet128()

    

    fid = load_real_fid_model('imagenet128', device=device)
    fid_final = fid.compute()
    print(f'Final FID: {fid_final:.4f}')
    output = dict(fid=fid_final, num_samples=num_processed_samples, beta_start=beta_start, beta_end=beta_end, num_timesteps=num_timesteps)
    with open(FID / 'imagenet128.json', 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    main()