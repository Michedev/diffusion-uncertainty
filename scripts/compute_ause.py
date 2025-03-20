import gc
import random
import torch
import numpy as np
import argparse

import yaml
from diffusion_uncertainty import pipeline_uncertainty
from diffusion_uncertainty import dataset
from diffusion_uncertainty.argparse import add_scheduler_uncertainty_args_
from diffusion_uncertainty.dataset import cifar10, imagenet
from diffusion_uncertainty.init_model import init_model_scheduler
from diffusion_uncertainty.metrics.ause import compute_aucs, compute_aucs_from_curve
from diffusion_uncertainty.paths import AUSE, CIFAR10, CONFIG, DIFFUSION_STARTING_POINTS
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty import DiffusionClassConditionalWithUncertainty
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered import DDIMSchedulerUncertainty
from diffusion_uncertainty.utils import DATASET_IMAGE_SIZE, load_X_T, load_y, predict_model
from scripts.generate_images_with_uncertainty_percentile import instantiate_uc_scheduler
import pytorch_lightning as pl
from tqdm import tqdm

def generate_halfway(X_0, y_batch, sampler, device, model):
    generation_steps = len(sampler.timesteps)
    X_0 = X_0.to(device)
    X_0 = X_0 * 2 - 1
    y_batch = y_batch.to(device)
    X_t = sampler.add_noise(X_0, torch.randn_like(X_0), timesteps=generation_steps//2)
    uncertainties = []
    scores = []
    intermediates = []
    sampler.prompt_embeds = y_batch
    with torch.no_grad():
        for t in sampler.timesteps[len(sampler.timesteps)//2:]:
            t = t.item()
            t_tensor = torch.full((X_t.shape[0],), t, device=device, dtype=torch.long)
            noisy_residual = predict_model(model, X_t, t_tensor, y_batch)
            output = sampler.step(noisy_residual, t, X_t)
            prev_noisy_sample = output.prev_sample
            if sampler.timestep_after_step >= t >= sampler.timestep_end_step:
                uncertainties.append(output.uncertainty.cpu())
                scores.append(output.pred_epsilon.cpu())
            X_t = prev_noisy_sample
        gen_images = (X_t / 2 + 0.5).clamp(0, 1)
        gen_images = gen_images * 255.0
        gen_images = gen_images.round()
        gen_images = gen_images.to(torch.uint8)
    return gen_images, torch.stack(uncertainties, dim=1), torch.stack(scores, dim=1)

from torch.utils.data import ConcatDataset 
def load_dataset(dataset_name: str):
    if dataset_name == 'cifar10':
        train_dataset = cifar10.CIFAR10Dataset(CIFAR10, 'train')
        test_dataset = cifar10.CIFAR10Dataset(CIFAR10, 'test')
        dataset = ConcatDataset([train_dataset, test_dataset])
    elif dataset_name.startswith('imagenet'):
        from diffusion_uncertainty import paths
        dataset_path = getattr(paths, dataset_name.upper())
        train_dataset = imagenet.ImagenetDataset(dataset_path, 'train')
        val_dataset = imagenet.ImagenetDataset(dataset_path, 'val')
        dataset = ConcatDataset([train_dataset, val_dataset])
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return dataset

def load_test_dataset(dataset_name: str):
    if dataset_name == 'cifar10':
        test_dataset = cifar10.CIFAR10Dataset(CIFAR10, 'test')
        
    elif dataset_name.startswith('imagenet'):
        from diffusion_uncertainty import paths
        dataset_path = getattr(paths, dataset_name.upper())
        test_dataset = imagenet.ImagenetDataset(dataset_path, 'val')
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return dataset

@torch.no_grad()
def main():
    args = parse_arguments()
    device = args.device
    image_size = DATASET_IMAGE_SIZE[args.dataset]
    scheduler_type: str = args.scheduler_type
    model, scheduler = init_model_scheduler(args.dataset)
    model = model.to(device)
    uc_scheduler = instantiate_uc_scheduler(args, scheduler, model, y=None)
    uc_scheduler.set_timesteps(args.num_steps_uc)
    
    dataset = load_dataset(args.dataset)

    print('loaded test dataset')

    assert len(dataset) >= args.num_samples, f'Number of samples {args.num_samples} is greater than the dataset size {len(dataset)}'

    # Set a fixed random seed for reproducibility
    pl.seed_everything(args.seed)

    
    # Create a random permutation of indices
    indices = torch.randperm(len(dataset))
    if args.num_samples is not None:
        indices = indices[:args.num_samples]
    
    # Use SubsetRandomSampler with the permuted indices
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler
    )

    ause_dataset = AUSE / args.dataset
    if not ause_dataset.exists():
        ause_dataset.mkdir()

    X_0_list = torch.zeros(len(dataset), 3, image_size, image_size, device='cpu', dtype=torch.float16)
    X_0_recon_list = torch.zeros(len(dataset), 3, image_size, image_size, device='cpu', dtype=torch.float16)
    uncertainty_list = torch.zeros(len(dataset), 3, image_size, image_size, device='cpu', dtype=torch.float16)
    print('Made X_0_list, X_0_recon_list, uncertainty_list')
    intervals = 50
    i_batch = 0
    for batch in (data_loader):
        image = batch['image']
        y = batch['label']
        slice_batch = slice(i_batch * args.batch_size, (i_batch) * args.batch_size + len(image))

        X_0_recon, uncertainty, _ = generate_halfway(image, y, uc_scheduler, device, model)
        uncertainty = uncertainty.sum(dim=1)
        X_0_list[slice_batch] = image.cpu().to(torch.float16)
        X_0_recon_list[slice_batch] = X_0_recon.cpu().to(torch.float16)
        uncertainty_list[slice_batch] = uncertainty.cpu().to(torch.float16)

        i_batch += 1
    if args.invert_uncertainty:
        uncertainty_list = -uncertainty_list
    mean_ause, mean_aurg = compute_aucs(X_0_list, X_0_recon_list, uncertainty_list, intervals=intervals)[0]['rmse']

    del X_0_list, X_0_recon_list, uncertainty_list
    del model, uc_scheduler
    torch.cuda.empty_cache()
    gc.collect()

    if isinstance(mean_ause, (torch.Tensor, np.ndarray)):
        mean_ause = mean_ause.item()
    if isinstance(mean_aurg, (torch.Tensor, np.ndarray)):
        mean_aurg = mean_aurg.item()
    print(f'Mean AUSE: {mean_ause}, Mean AURG: {mean_aurg}', flush=True)

    result_dict = {
        'mean_ause': str(mean_ause),
        'mean_aurg': str(mean_aurg),
    }

    print('Results:', result_dict, flush=True)
    suffix = '_inverted' if args.invert_uncertainty else ''
    with open(ause_dataset / f'results_{scheduler_type}{suffix}.yaml', 'w') as f:
        yaml.dump(result_dict, f)
    with open(ause_dataset / f'args.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)

    return 

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", "--dataset-name", type=str, dest="dataset")
    argparser.add_argument('--batch-size', type=int, default=32, dest='batch_size')
    argparser.add_argument('--device', type=str, default='cuda', dest='device')
    argparser.add_argument('--num-samples', type=int, default=None, dest='num_samples')
    argparser.add_argument('--invert-uncertainty', '--invert', action='store_true', dest='invert_uncertainty')

    add_scheduler_uncertainty_args_(argparser)

    argparser.add_argument('--config', type=str, help='path to the config file')
    
    args = argparser.parse_args()

    if args.config is not None:
        config_ause = CONFIG / 'ause' / args.config
        with open(config_ause, 'r') as f:
            config = yaml.safe_load(f)
        argparser.set_defaults(**config)
        args = argparser.parse_args()

    return args


if __name__ == "__main__":
    main()