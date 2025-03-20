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
import yaml
from torch.multiprocessing.spawn import ProcessExitedException

from diffusion_uncertainty.schedulers_uncertainty.get_uncertainty_scheduler import get_uncertainty_scheduler
from diffusion_uncertainty.utils import get_generation_configs, load_config
from diffusion_uncertainty.uvit.load_pretrained_models import load_autoencoder_uvit, load_uvit, load_uvit_scheduler
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusion_uncertainty.argparse import add_scheduler_uncertainty_args_

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
def generate_and_save(gpu_idx: int, args, num_samples: int, dest_folder_datetime: Path):

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

    print(f'{uc_scheduler.M=} {uc_scheduler.after_step=} {uc_scheduler.num_steps_uc=}')
    uc_scheduler.set_timesteps(args.generation_steps)
    with torch.autocast(device_type='cuda'):
        intermediates = generate_samples_model_scheduler_unconditioned_from_tensor(X_T, args.batch_size, model=unet, scheduler=uc_scheduler, device=device)
    print('intermediates', intermediates.keys())

    torch.save(uc_scheduler.timesteps, dest_folder_datetime / f'timestep_{gpu_idx}.pth')
    torch.save(intermediates['score'], dest_folder_datetime / 'score.pth')
    torch.save(intermediates['uncertainty'], dest_folder_datetime / f'uncertainty_{args.scheduler_type}_{gpu_idx}.pth')
    torch.save(intermediates['gen_images'], dest_folder_datetime / f'gen_images_{gpu_idx}.pth')


@torch.no_grad()
def main():

    args = parse_args()
    args.model_type = 'unet'
    print('Set model type to unet')

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

    dest_folder = RESULTS / 'score-uncertainty'
    if not dest_folder.exists(): dest_folder.mkdir()

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_folder_datetime = dest_folder / datetime_str

    if not dest_folder_datetime.exists():
        dest_folder_datetime.mkdir()

    args.__dict__['dataset'] = f'cifar10'
    
    with open(dest_folder_datetime /  'args.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

        f.write('\n\n')
        f.write('# if the filename is uncertainty_{gpu_idx}.pth, then the uncertainty is computed using the mc_dropout scheduler\n')    


    device = torch.device('cuda')

    num_gpus = torch.cuda.device_count()
    print('num_gpus', num_gpus)
    try:  
        torch.multiprocessing.spawn(
            fn=generate_and_save,
            args=(args, args.num_samples // num_gpus, dest_folder_datetime),
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