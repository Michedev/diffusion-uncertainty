"""
Generate dataset of score and uncertainty for imagenet64.
The dataset is saved in a folder in RESULTS/score-uncertainty/{datetime}/
The dataset is saved as a dictionary with keys:
    - y: the class of the generated image
    - timestep: the timestep of the generated image
    - score: the score of the generated image
    - uncertainty: the uncertainty of the generated image
"""
import json
import sys
import timeit
from path import Path
import yaml

from diffusion_uncertainty.utils import load_config
from diffusion_uncertainty.uvit.autoencoder import FrozenAutoencoderKL
from diffusion_uncertainty.uvit.load_pretrained_models import load_autoencoder_uvit, load_uvit, load_uvit_scheduler
from diffusion_uncertainty.uvit.uvit import UViT
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusion_uncertainty.schedulers_uncertainty.get_uncertainty_scheduler import get_uncertainty_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler

sys.path.insert(0, Path(__file__).absolute().parent.parent)

import datetime
from diffusion_uncertainty.generate_samples import generate_samples_model_scheduler_class_conditioned, generate_samples_model_scheduler_class_conditioned_from_tensor, generate_samples_model_scheduler_class_conditioned_uvit, generate_samples_model_scheduler_class_conditioned_uvit_from_tensor
from diffusion_uncertainty.init_model import init_guided_diffusion_imagenet64_and_scheduler, init_guided_diffusion_imagenet128_and_scheduler
from diffusion_uncertainty.paths import CONFIG, DIFFUSION_STARTING_POINTS, RESULTS
import torch
import argparse
import pytorch_lightning as pl
import gzip

@torch.no_grad()
def generate_and_save(gpu_idx: int, args: argparse.Namespace, num_samples: int, dest_folder_datetime: Path):

    # if num_samples % args.batch_size != 0:
    #     num_samples = num_samples + (args.batch_size - num_samples % args.batch_size)
    #     print('num_samples is not divisible by batch_size, setting num_samples to', args.num_samples)

    pl.seed_everything(args.seed)

    device = torch.device('cuda', gpu_idx)

    # fix name in case of unet
    model_type = args.model_type
    if model_type == 'unet':
        model_type = 'adm'
    suffix = ''
    if args.image_size == 128 and args.model_type == 'uvit':
        suffix = '_uvit'

    index_range = slice(args.start_index + gpu_idx * num_samples, args.start_index + (gpu_idx + 1) * num_samples)

    y = torch.load(DIFFUSION_STARTING_POINTS / f'imagenet{args.image_size}{suffix}' / 'y.pth', map_location='cpu')[index_range]
    y = y.to(device)

    X_T = torch.load(DIFFUSION_STARTING_POINTS / f'imagenet{args.image_size}{suffix}' / 'X_T.pth', map_location='cpu')[index_range]
    X_T = X_T.to(device)

    if args.model_type == 'unet':
        if args.image_size == 64:
            unet, scheduler = init_guided_diffusion_imagenet64_and_scheduler(dropout=args.dropout)
        elif args.image_size == 128:
            unet, scheduler = init_guided_diffusion_imagenet128_and_scheduler()
        else:
            raise NotImplementedError("Only image_size=64 and image_size=128 are supported for unet model type.")
    else:

        if args.image_size == 256:
            uvit: UViT = load_uvit(256, 'cuda')
        elif args.image_size == 512:
            uvit = load_uvit(512, 'cuda')
        else:
            raise NotImplementedError("Only image_size=256 and image_size=512 are supported for uvit model type.")
        autoencoder: FrozenAutoencoderKL = load_autoencoder_uvit('cuda')
        unet = UViTAE(uvit, autoencoder)
        scheduler = load_uvit_scheduler()
    unet = unet.eval()
    unet.to(device)

    uc_scheduler = get_uncertainty_scheduler(args, y, unet, scheduler)

    uc_scheduler.set_timesteps(args.generation_steps)
    if args.model_type == 'unet':
        start_time = timeit.default_timer()
        with torch.autocast(device_type='cuda'):
            intermediates = generate_samples_model_scheduler_class_conditioned_from_tensor(X_T, y, args.batch_size, model=unet, scheduler=uc_scheduler, device=device)
        end_time = timeit.default_timer()
    else:
        start_time = timeit.default_timer()
        with torch.autocast(device_type='cuda'):
            intermediates = generate_samples_model_scheduler_class_conditioned_uvit_from_tensor(X_T, y, args.batch_size, uvit_ae=unet, scheduler=uc_scheduler, device=device)
        end_time: float = timeit.default_timer()
    delta_uc = end_time - start_time
    print('Time Uncertainty estimation', delta_uc)
    del intermediates

    if (args.image_size == 128 and args.model_type == 'uvit') or (args.image_size > 128):
        scheduler = KDPM2DiscreteScheduler.from_config(uc_scheduler.config)
    else:
        scheduler = DDIMScheduler.from_config(uc_scheduler.config)
    scheduler.set_timesteps(args.generation_steps)


    if args.model_type == 'unet':
        start_time = timeit.default_timer()
        with torch.autocast(device_type='cuda'):
            intermediates = generate_samples_model_scheduler_class_conditioned_from_tensor(X_T, y, args.batch_size, model=unet, scheduler=scheduler, device=device)
        end_time = timeit.default_timer()
    else:
        start_time = timeit.default_timer()
        with torch.autocast(device_type='cuda'):
            intermediates = generate_samples_model_scheduler_class_conditioned_uvit_from_tensor(X_T, y, args.batch_size, uvit_ae=unet, scheduler=scheduler, device=device)
        end_time: float = timeit.default_timer()
    delta_normal = end_time - start_time

    print('Time normal', delta_normal)
    
    print('------------------')
    print('Time Uncertainty estimation', delta_uc)
    print('Time normal', delta_normal)
    times_path = RESULTS / 'times.json'
    times = []
    if times_path.exists():
        with open(times_path) as f:
            times = json.load(f)
    results = dict(dataset='imagenet', image_size=args.image_size, time_uc=delta_uc, time_normal=delta_normal, model_type=model_type, scheduler_type=args.scheduler_type)
    print('results', results)
    times.append(results)
    with open(times_path, 'w') as f:
        json.dump(times, f)
    # print('intermediates', intermediates.keys())

@torch.no_grad()
def main():

    args = parse_args()

    if args.scheduler_type == 'mc_dropout':
        assert args.image_size == 64, 'mc_dropout only implemented for image_size=64'
    pl.seed_everything(args.seed)

    dest_folder = RESULTS / 'score-uncertainty'
    if not dest_folder.exists(): dest_folder.mkdir()

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_folder_datetime = dest_folder / datetime_str

    if not dest_folder_datetime.exists():
        dest_folder_datetime.mkdir()

    # if args.num_samples % args.batch_size != 0:
    #     args.num_samples = args.num_samples + (args.batch_size - args.num_samples % args.batch_size)
    #     print('num_samples is not divisible by batch_size, setting num_samples to', args.num_samples)

    pl.seed_everything(args.seed)


    args.__dict__['dataset'] = f'imagenet{args.image_size}'  # todo: change this to a argparse argument in the future


    num_gpus = torch.cuda.device_count()
    print('num_gpus', num_gpus)
    torch.multiprocessing.spawn(
        fn=generate_and_save,
        args=(args, args.num_samples // num_gpus, dest_folder_datetime),
        nprocs=num_gpus,
        join=True,
    )
    print('Done')

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--num-samples', type=int, default=300, dest='num_samples')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--generation-steps', type=int, default=20, dest='generation_steps')
    argparser.add_argument('-M', type=int, default=30, dest='M')
    argparser.add_argument('--start-step-uc', '--start-step', type=int, default=0, dest='start_step_uc')
    argparser.add_argument('--num-steps-uc', type=int, default=20, dest='num_steps_uc')
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--eta', type=float, default=0.00)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--class-condition')
    argparser.add_argument('--image-size', type=int, choices=[64, 128, 256, 512], dest='image_size', default=None)
    argparser.add_argument('--scheduler-type', '--scheduler', type=str, default=None, choices=['mc_dropout', 'flip', 'uncertainty', 'flip_grad', 'uncertainty_single', 'uncertainty_single_score', 'uncertainty_centered', 'uncertainty_centered_d', 'uncertainty_image', 'uncertainty_original', 'uncertainty_zigzag_centered', 'uncertainty_fisher', 'dpm_2_uncertainty_centered'], dest='scheduler_type')
    argparser.add_argument('--start-index', type=int, default=None, dest='start_index')
    argparser.add_argument('--multi-gpu', action='store_true', default=False)
    argparser.add_argument('--model-type', type=str, default=None, choices=['unet', 'uvit'], dest='model_type')
    argparser.add_argument('--measure-time', action='store_true', dest='measure_time')

    uncertainty_args = argparser.add_argument_group('uncertainty')
    uncertainty_args.add_argument('--predict-next', action='store_true', dest='predict_next')

    infere_noise_distance_args = argparser.add_argument_group('uncertainty_distance')
    infere_noise_distance_args.add_argument('--uncertainty-distance', type=int, default=20, dest='uncertainty_distance')

    uncertainty_zigzag_centered_args = argparser.add_argument_group('uncertainty_zigzag_centered')
    uncertainty_zigzag_centered_args.add_argument('--num-zigzag', '--num-zigzags', '--num-zig-zag', '--num-zig-zags', type=int, default=3, dest='num_zigzag')

    options = CONFIG.joinpath('generation').files('*.yaml')
    options = [x.basename().replace('.yaml', '') for x in options]

    argparser.add_argument('--config', type=str, default=None, choices=options, help='path to the config file')

    args = argparser.parse_args()

    if args.config is not None:
        print('Loading config file')
        args = load_config(CONFIG / 'generation', args.config)
        print('Loaded config file')
        argparser.set_defaults(**args.__dict__)
        print('Set defaults from config file')
        args = argparser.parse_args()
    else:
        assert args.image_size is not None, '--image-size must be specified'
        assert args.scheduler_type is not None, '--scheduler-type must be specified'
        # assert args.model_type is not None, '--model-type must be specified'
        assert args.start_index is not None, '--start-index must be specified'

    return args
    
if __name__ == '__main__':
    main()