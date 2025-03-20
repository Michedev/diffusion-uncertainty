"""
Generate dataset of score and uncertainty for imagenet64.
The dataset is saved in a folder in RESULTS/score-uncertainty/{datetime}/
The dataset is saved as a dictionary with keys:
    - y: the class of the generated image
    - timestep: the timestep of the generated image
    - score: the score of the generated image
    - uncertainty: the uncertainty of the generated image
"""
import sys
from path import Path
import json
import timeit
import yaml
from torch.multiprocessing.spawn import ProcessExitedException

from diffusion_uncertainty.schedulers_uncertainty.get_uncertainty_scheduler import get_uncertainty_scheduler
from diffusion_uncertainty.utils import get_generation_configs, load_config
from diffusion_uncertainty.uvit.load_pretrained_models import load_autoencoder_uvit, load_uvit, load_uvit_scheduler
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusion_uncertainty.argparse import add_scheduler_uncertainty_args_
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

sys.path.insert(0, Path(__file__).absolute().parent.parent)

import pytorch_lightning as pl
import datetime
from diffusion_uncertainty.generate_samples import generate_samples_model_scheduler_class_conditioned, generate_samples_model_scheduler_class_conditioned_from_tensor, generate_samples_model_scheduler_class_conditioned_uvit, generate_samples_model_scheduler_class_conditioned_uvit_from_tensor, generate_samples_model_scheduler_unconditioned_from_tensor
from diffusion_uncertainty.init_model import init_ddpm_cifar_10_and_scheduler, init_guided_diffusion_imagenet64_and_scheduler, init_guided_diffusion_imagenet128_and_scheduler
from diffusion_uncertainty.paths import CONFIG, DIFFUSION_STARTING_POINTS, RESULTS
import torch
import argparse
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyCifar10 as DDIMSchedulerUncertaintyUncertainty
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout import DDIMSchedulerUncertaintyCifar10 as DDIMSchedulerUncertaintyImagenetMCDropout
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered import DDIMSchedulerUncertaintyCifar10 as DDIMSchedulerUncertaintyUncertaintyCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_zigzag import DDIMSchedulerUncertaintyCifar10  as DDIMSchedulerUncertaintyUncertaintyZigZag


@torch.no_grad()
def generate_and_save(gpu_idx: int, args, num_samples: int):

    pl.seed_everything(args.seed)

    device = torch.device('cuda', gpu_idx)

    # fix name in case of unet
    model_type = args.model_type
    if model_type == 'unet':
        model_type = 'adm'

    index_range = slice(args.start_index + gpu_idx * num_samples, args.start_index + (gpu_idx + 1) * num_samples)

    y = torch.load(DIFFUSION_STARTING_POINTS / 'cifar10' / 'y.pth', map_location='cpu')[index_range]
    y = y.to(device)

    X_T = torch.load(DIFFUSION_STARTING_POINTS / 'cifar10' / 'X_T.pth', map_location='cpu')[index_range]
    X_T = X_T.to(device)

    unet, scheduler = init_ddpm_cifar_10_and_scheduler(dropout=args.dropout) 
    unet = unet.eval()
    unet.to(device)

    assert unet is not None, 'unet is None'
    assert scheduler is not None, 'scheduler is None'


    uc_scheduler = get_uncertainty_scheduler(args, y, unet, scheduler)

    uc_scheduler.set_timesteps(args.generation_steps)
    start_time = timeit.default_timer()
    with torch.autocast(device_type='cuda'):
        intermediates = generate_samples_model_scheduler_unconditioned_from_tensor(X_T, args.batch_size, model=unet, scheduler=uc_scheduler, device=device)
    end_time = timeit.default_timer()
    delta_uc = end_time - start_time
    print('Time Uncertainty estimation', delta_uc)
    del intermediates

    scheduler = DDIMScheduler.from_config(uc_scheduler.config)
    scheduler.set_timesteps(args.generation_steps)


    start_time = timeit.default_timer()
    with torch.autocast(device_type='cuda'):
        intermediates = generate_samples_model_scheduler_unconditioned_from_tensor(X_T, args.batch_size, model=unet, scheduler=scheduler, device=device)
    end_time = timeit.default_timer()
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
    results = dict(dataset='cifar10', image_size=32, time_uc=delta_uc, time_normal=delta_normal, model_type=model_type, scheduler_type=args.scheduler_type)
    print('results', results)
    times.append(results)
    with open(times_path, 'w') as f:
        json.dump(times, f, indent=4)
    # print('intermediates', intermediates.keys())



@torch.no_grad()
def main():

    args = parse_args()
    args.model_type = 'unet'
    print('Set model type to unet')

    pl.seed_everything(args.seed)

    dest_folder = RESULTS / 'score-uncertainty'
    if not dest_folder.exists(): dest_folder.mkdir()


    args.__dict__['dataset'] = f'cifar10'
    

    device = torch.device('cuda')

    num_gpus = torch.cuda.device_count()
    print('num_gpus', num_gpus)
    try:  
        torch.multiprocessing.spawn(
            fn=generate_and_save,
            args=(args, args.num_samples // num_gpus),
            nprocs=num_gpus,
            join=True,
            
        )
    except ProcessExitedException as e:
        print(f"A child process exited with non-zero exit code: {e.exit_code}")
        print("Error message subprocess:", e.msg)
    print('Done')

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('--num-samples', type=int, default=300, dest='num_samples')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--generation-steps', type=int, default=20, dest='generation_steps')
    
    add_scheduler_uncertainty_args_(argparser)



    configs = get_generation_configs('generation')

    argparser.add_argument('--config', type=str, help='path to the config file', choices=configs)

    args = argparser.parse_args()

    if args.config is not None:
        print('Loading config file - ignoring other arguments')
        args = load_config(CONFIG / 'generation', args.config)
    else:
        assert args.start_index is not None, 'start_index is required'
    return args

if __name__ == '__main__':
    main()