from random import choice
from unittest import result
from cv2 import threshold
from path import Path
import sys

from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional import DiffusionClassConditional
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty import DiffusionClassConditionalWithUncertainty
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty_guided_gradient import DiffusionClassConditionalGuidedGradient
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty_guided_posterior_distribution import DiffusionClassConditionalGuidedPosteriorDistribution
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty_guided_second_order import DiffusionClassConditionalGuidedSecondOrder
from diffusion_uncertainty.pipeline_uncertainty.uncertainty_guidance import generate_samples_model_scheduler_class_conditioned_with_threshold
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import datetime
import gzip
import json
import numpy as np
from omegaconf import OmegaConf
from pytorch_fid.inception import InceptionV3
import torch
import yaml
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from diffusion_uncertainty.fid import compute_fid_score_bayesdiff_from_dataset, compute_fid_score_torchmetrics, load_real_fid_model
from diffusion_uncertainty.generate_samples import generate_samples_model_scheduler_class_conditioned, generate_samples_model_scheduler_class_conditioned_with_percentile, generate_samples_uvit_scheduler_class_conditioned_with_threshold
from diffusion_uncertainty.guided_diffusion.unet_openai import UNetModel
from diffusion_uncertainty.init_model import instantiate_model_scheduler
from diffusion_uncertainty.paths import CONFIG, FID, RESULTS, ROOT, THRESHOLD, DIFFUSION_STARTING_POINTS
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlip
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertainty
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_single import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintySingle
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip_grad import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipGrad
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_image import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyImage
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_single_score import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintySingleScore
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered_d import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCenteredD
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_zigzag_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyZigZagCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyOriginal
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout
from diffusion_uncertainty.utils import load_config
from diffusion_uncertainty.uvit.autoencoder import FrozenAutoencoderKL
from diffusion_uncertainty.uvit.uvit import UViT
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.unets import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from scripts.compute_dataset_fid import calculate_activation_statistics_dataset, calculate_frechet_distance, instantiate_inception_v3
from torchvision.utils import make_grid
from torchvision.utils import save_image
from diffusion_uncertainty.schedulers_uncertainty.get_uncertainty_scheduler import instatiate_uncertainty_scheduler

dataset_model_map = {
    'imagenet64': ROOT.joinpath('models/64x64_diffusion.pth'),
    'imagenet128': ROOT.joinpath('models/128x128_diffusion.pth'),
}


def compute_fid_score_gen_images_guidance(path_storage_images: Path, device = torch.device('cuda'), batch_size = 64):
    gen_images = torch.load(path_storage_images / 'gen_images.pth')
    gen_images = gen_images.to(device)
    gen_images_guidance = torch.load(path_storage_images / 'gen_images_threshold.pth')
    gen_images_guidance = gen_images_guidance.to(device)
    print('Shape of gen_images:', gen_images.shape)
    print('Shape of gen_images_guidance:', gen_images_guidance.shape)
    if gen_images.dtype == torch.uint8:
        print('Converting gen_images to float')
        gen_images = gen_images.float() / 255.0
    if gen_images_guidance.dtype == torch.uint8:
        print('Converting gen_images_guidance to float')
        gen_images_guidance = gen_images_guidance.float() / 255.0
    with open(path_storage_images / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset']
    fid_score = compute_fid_score_bayesdiff_from_dataset(gen_images=gen_images, dataset_name=dataset_name, device=device, batch_size=batch_size)
    fid_score_guidance = compute_fid_score_bayesdiff_from_dataset(gen_images=gen_images_guidance, dataset_name=dataset_name, device=device, batch_size=batch_size)
    return fid_score, fid_score_guidance


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num-samples', type=int, default=1000, help='number of samples to generate')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--scheduler-type', '--scheduler', dest='scheduler_type', type=str, default='uncertainty')
    argparser.add_argument('--num-steps', default=20, type=int, help='number of steps for the diffusion used when use_percentile is not set', dest='num_steps')
    argparser.add_argument('--dataset-folder', '--dataset', type=str, required=False, help='path to the dataset folder containing the uncertainty and gen_images files', dest='dataset_name')
    argparser.add_argument('--start-step-threshold', '--start-step', '--start-step-guidance',  type=int, default=0, help='step to start estimating the threshold', dest='start_step_guidance')
    argparser.add_argument('--num-steps-threshold', '--num-steps-uc', '--num-steps-guidance', type=int, default=20, help='number of steps to estimate the threshold', dest='num_steps_guidance')
    argparser.add_argument('--start-index', type=int, default=0, help='starting index of the samples')
    argparser.add_argument('--guidance-type', type=str, default='gradient', help='type of guidance to use', dest='guidance_type', choices=['gradient', 'posterior', 'second_order'])
    argparser.add_argument('--seed', type=int, default=491, help='random seed')
    argparser.add_argument('--percentile', type=float, default=0.95, help='percentile for the threshold', dest='percentile')
    argparser.add_argument('--skip-save', action='store_true', help='skip saving the generated images', dest='skip_save')
    argparser.add_argument('--checkpoint', '--checkpoint-path', type=str, help='path to the model checkpoint', default=None, dest='checkpoint_path')
    argparser.add_argument('--use-percentile', action='store_true', help='use the percentile instead of saved threshold')
    argparser.add_argument('--skip-ddim', action='store_true', help='skip the DDIM sampling')
    argparser.add_argument('--gradient-type', '--wrt', '--gradient-wrt', type=str, default='input', help='type of gradient to use', choices=['input', 'score'], dest='gradient_wrt')
    argparser.add_argument('--gradient-direction', type=str, choices=['ascend', 'descend'], default='descend', help='direction of the gradient', dest='gradient_direction')
    argparser.add_argument('--lambda', '--lambda-update', type=float, default=0.1, help='lambda for updating the threshold', dest='lambda_update')
    argparser.add_argument('--threshold-type', type=str, choices=['higher', 'lower'], default='higher', help='type of threshold to use i.e. if affect pixels with uncertainty lower than a percentile or higher', dest='threshold_type')

    
    argparser.add_argument('--config', type=str, help='path to the config file')

    
    args = argparser.parse_args()

    if args.config is not None:
        print('Loading config file - ignoring other arguments')
        args_config = load_config(CONFIG / 'guidance', args.config)
        # this gives priority to the terminal arguments over config file
        argparser.set_defaults(**args_config.__dict__)
        args = argparser.parse_args()

    return args




def main():
    args = parse_args()

    print('Arguments:', args.__dict__)

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')
    seed = args.seed

    pl.seed_everything(seed)

    if args.checkpoint_path is not None:
        checkpoint_path = RESULTS / args.checkpoint_path
        if not checkpoint_path.exists():
            # Check if the checkpoint path is in the uncertainty_guidance folder
            for folder in RESULTS.joinpath('uncertainty_guidance').walkdirs():
                if folder.basename() == args.checkpoint_path:
                    checkpoint_path = folder
                    break
        assert checkpoint_path.exists(), f'Checkpoint path {checkpoint_path} does not exist'
        print(f'Using checkpoint from {checkpoint_path}')
        fid_score, fid_score_guidance = compute_fid_score_gen_images_guidance(checkpoint_path)
        print(f'FID score: {fid_score}')
        print(f'FID score with guidance: {fid_score_guidance}')
        return

    model, scheduler = instantiate_model_scheduler(args.dataset_name)
    
    model: UNetModel | UViTAE = model.to(device)
    if args.dataset_name == 'cifar10':
        image_size = 32
    else:
        image_size = int(args.dataset_name.replace('imagenet', ''))
    # fid_model = load_real_fid_model(args.dataset_name, device)
    if not args.use_percentile:
        thresholds = torch.load(THRESHOLD / args.dataset_name / f'thresholds_{args.scheduler_type}_perc={args.percentile}.pth', map_location=device)
    else:
        thresholds = args.percentile
        

    # generator = torch.Generator(device=device).manual_seed(seed)
    # y = torch.randint(0, 1000, (args.num_samples,), generator=generator, device=device)

    range_starting_points = slice(args.start_index, args.start_index + args.num_samples)



    if not args.use_percentile:
        with open(Path(THRESHOLD / args.dataset_name / f'config_{args.scheduler_type}_perc={args.percentile}.yaml'), 'r') as f:
            config_threshold = yaml.safe_load(f)
        config_generation = config_threshold['dataset_config']

        generation_steps = config_generation['generation_steps']
        print(config_generation.keys())

        threshold_folder = args.dataset_name
        model_type = config_generation['model_type']
        if model_type == 'uvit':
            threshold_folder += '_uvit'
        elif model_type == 'unet':
            threshold_folder += '_adm'
        else:
            raise ValueError(f'Unknown model type {model_type}')
        for k, v in config_generation.items():
            if k not in args.__dict__:
                args.__dict__[k] = v
                setattr(args, k, v)
        if 'start_step_uc' not in args.__dict__:
            args.start_step_uc = 0
        if 'num_steps_uc' not in args.__dict__:
            args.num_steps_uc = generation_steps
    generation_steps = args.num_steps

    x_T = torch.load(DIFFUSION_STARTING_POINTS / args.dataset_name / 'X_T.pth', map_location=device)[range_starting_points]
    y = torch.load(DIFFUSION_STARTING_POINTS / args.dataset_name / 'y.pth', map_location=device)[range_starting_points]





    ddim_sampler = DDIMScheduler.from_config(config=scheduler.config)
    ddim_sampler.set_timesteps(generation_steps)

    # uc_scheduler = instatiate_uncertainty_scheduler(args, y, model, scheduler)
    # uc_scheduler.set_timesteps(generation_steps)

    pipeline_sampler_ddim = DiffusionClassConditional(model, ddim_sampler, image_size, device, args.batch_size, seed)

    if args.guidance_type == 'gradient':
        print('Using gradient guidance')
        pipeline_sampler_threshold = DiffusionClassConditionalGuidedGradient(model, ddim_sampler, thresholds, image_size, device, args.batch_size, seed, gradient_wrt=args.gradient_wrt, threshold_type=args.threshold_type, gradient_direction=args.gradient_direction, lambda_update=args.lambda_update)
    elif args.guidance_type == 'posterior':
        print('Using posterior distribution guidance')
        pipeline_sampler_threshold = DiffusionClassConditionalGuidedPosteriorDistribution(model, ddim_sampler, thresholds, image_size, device, args.batch_size, seed, threshold_type=args.threshold_type)
    elif args.guidance_type == 'second_order':
        print('Using second order guidance')
        pipeline_sampler_threshold = DiffusionClassConditionalGuidedSecondOrder(model, ddim_sampler, thresholds, image_size, device, args.batch_size, seed, threshold_type=args.threshold_type)
    else:
        raise ValueError(f'Unknown guidance type {args.guidance_type}')


    with torch.autocast('cuda'):
        ddim_sampler.set_timesteps(generation_steps // 4)

        if not args.skip_ddim:
            output = pipeline_sampler_ddim(X_T=x_T, y=y)
        print(f'using {generation_steps//4} steps for DDIM with guidance')
        ddim_sampler.set_timesteps(generation_steps // 4)
        output_threshold = pipeline_sampler_threshold(X_T=x_T, y=y, start_step=args.start_step_guidance, num_steps=args.num_steps_guidance)

    config = ({ 
        'dataset': args.dataset_name,
        'scheduler_type': args.scheduler_type,
        'num_samples': args.num_samples,
        'seed': seed,
        'start_step_threshold': args.start_step_guidance,
        'num_steps_threshold': args.num_steps_guidance,
        'start_index': args.start_index,
        'percentile': args.percentile,
        'use_percentile': args.use_percentile,
        'guidance_type': args.guidance_type,
    })

    if not args.skip_save and not args.skip_ddim:
        
        gen_images_threshold = output_threshold['gen_images']
        gen_images = output['gen_images']

        storage_images = RESULTS / 'uncertainty_guidance'
        if not storage_images.exists():
            storage_images.mkdir()

        storage_images = storage_images / f'{args.dataset_name}_{args.scheduler_type}_perc={args.percentile}'
        storage_images = storage_images / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not storage_images.exists():
            storage_images.makedirs()
        
        with open(storage_images / 'args.yaml', 'w') as f:
            yaml.safe_dump(args.__dict__, f)
        with open(storage_images / 'config.yaml', 'w') as f:
            yaml.safe_dump(config, f)

        torch.save(gen_images_threshold, storage_images / 'gen_images_threshold.pth')
        torch.save(gen_images, storage_images / 'gen_images.pth')

        print('Saved config to', storage_images / 'args.yaml')
        print('Saved gen_images_threshold to', storage_images / 'gen_images_threshold.pth')
        print('Saved gen_images to', storage_images / 'gen_images.pth')

        fid_score = compute_fid_score_bayesdiff_from_dataset(gen_images=output['gen_images'], dataset_name=args.dataset_name, device=device, batch_size=args.batch_size)
        fid_score_guidance = compute_fid_score_bayesdiff_from_dataset(gen_images=output_threshold['gen_images'], dataset_name=args.dataset_name, device=device, batch_size=args.batch_size)
        # fid_score, fid_score_guidance = compute_fid_score_gen_images_guidance(storage_images)
        print(f'FID score: {fid_score}')
        print(f'FID score with guidance: {fid_score_guidance}')
        
        results_guidance = dict(fid_score=fid_score, fid_score_guidance=fid_score_guidance, dataset=args.dataset_name, scheduler_type=args.scheduler_type, num_samples=args.num_samples, seed=seed, start_step_threshold=args.start_step_guidance, num_steps_threshold=args.num_steps_guidance, start_index=args.start_index, percentile=args.percentile, use_percentile=args.use_percentile)
    else:
        storage_images = RESULTS / 'uncertainty_guidance'

        results_guidance = dict(dataset=args.dataset_name, scheduler_type=args.scheduler_type, num_samples=args.num_samples, seed=seed, start_step_threshold=args.start_step_guidance, num_steps_threshold=args.num_steps_guidance, start_index=args.start_index, percentile=args.percentile, use_percentile=args.use_percentile)
        if not args.skip_ddim:
            fid_score: float = compute_fid_score_bayesdiff_from_dataset(gen_images=output['gen_images'], dataset_name=args.dataset_name, device=device, batch_size=args.batch_size)
            print(f'FID score: {fid_score}')
            results_guidance['fid_score'] = fid_score
            batch_gen_images = output['gen_images'][:64]
            if batch_gen_images.dtype == torch.uint8:
                batch_gen_images = batch_gen_images.float() / 255.0
            save_image(batch_gen_images, fp=storage_images / 'gen_images.png', nrow=8)
        fid_score_guidance = compute_fid_score_bayesdiff_from_dataset(gen_images=output_threshold['gen_images'], dataset_name=args.dataset_name, device=device, batch_size=args.batch_size)
        print(f'FID score with guidance: {fid_score_guidance}')
        results_guidance['fid_score_guidance'] = fid_score_guidance

        batch_gen_images_threshold = output_threshold['gen_images'][:64]
        if batch_gen_images_threshold.dtype == torch.uint8:
            batch_gen_images_threshold = batch_gen_images_threshold.float() / 255.0
        save_image(batch_gen_images_threshold, storage_images / 'gen_images_threshold.png', nrow=8)
    results_path = RESULTS / 'uncertainty_guidance' / 'results.json'
    if not results_path.exists():
        results_fid = []
    else:
        with open(results_path, 'r') as f:
            results_fid = json.load(f)
    results_fid.append(results_guidance)
    with open(results_path, 'w') as f:
        json.dump(results_fid, f, indent=4)

if __name__ == '__main__':
    main()