from sched import scheduler
from diffusion_uncertainty.init_model import init_scheduler
from diffusion_uncertainty.utils import resolve_dataset_folders_path
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math

import sys
from path import Path
# sys.path.append(Path(__file__).absolute().parent.parent)
from tqdm import tqdm
import yaml

from diffusion_uncertainty.paths import RESULTS, SCORE_DATASET_FID
from diffusion_uncertainty.utils import predict_batch_fid_vector
from scripts.compute_dataset_fid import calculate_frechet_distance

from diffusion_uncertainty.fid import get_dims_bayesdiff, load_inception_model_bayesdiff, load_real_fid_model
import argparse
import json
from torch.nn import functional as F


def compute_fid_torchmetrics(config, gen_images, uncertainties, args, device):
    fid_model = load_real_fid_model(config['dataset'], device=device)

    # Add padding to gen_images
    # padding = nn.ZeroPad2d(((224-gen_images.shape[-1])//2, (224-gen_images.shape[-1])//2, (224-gen_images.shape[-2])//2, (224-gen_images.shape[-2])//2))
    # gen_images = padding(gen_images)

    print(f'{gen_images.shape=}')
    print(f'{gen_images.dtype=}')


    assert uncertainties.shape[0] == gen_images.shape[0], f"{uncertainties.shape=} {gen_images.shape=}"

    perc = args.perc  # Percentage of samples to remove

    i_sorted = torch.argsort(uncertainties)
    if args.mode == 'max':
        for i in tqdm(range(math.ceil(len(uncertainties) * (1 - perc) // args.batch_size))):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            fid_model.update(gen_images_batch, real=False)
        mode_text = 'highest'
    elif args.mode == 'min':
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc) // args.batch_size), len(uncertainties) // args.batch_size)):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            fid_model.update(gen_images_batch, real=False)
        mode_text = 'lowest'
    elif args.mode == 'both':
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc / 2) // args.batch_size), math.ceil(len(uncertainties) * (1 - (perc / 2)) // args.batch_size))):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            fid_model.update(gen_images_batch, real=False)
        mode_text = 'outside the extreme'
    
    fid_score_uncertainty = fid_model.compute().item()

    print(f"fid score without the top {int(perc*100)}% samples with {mode_text} uncertainty: {fid_score_uncertainty:.3f}")

    del fid_model

    fid_model = load_real_fid_model(config['dataset'], device=device)

    i_random = torch.randperm(len(uncertainties))

    for i in tqdm(range(math.ceil(len(uncertainties) * (1 - perc) // args.batch_size))):
        i_batch = i_random[i * args.batch_size: (i + 1) * args.batch_size]
        gen_images_batch = gen_images[i_batch]
        gen_images_batch = gen_images_batch.to(device)
        fid_model.update(gen_images_batch, real=False)

    fid_score_random = fid_model.compute().item()

    return fid_score_uncertainty, fid_score_random

def compute_fid_score_bayesdiff(config, gen_images, uncertainties, args, device):
    dims = get_dims_bayesdiff(config['dataset'])
    model = load_inception_model_bayesdiff(device=device, dims=dims)
    model.eval()

    # Add padding to gen_images
    # padding = nn.ZeroPad2d(((224-gen_images.shape[-1])//2, (224-gen_images.shape[-1])//2, (224-gen_images.shape[-2])//2, (224-gen_images.shape[-2])//2))
    # gen_images = padding(gen_images)

    print(f'{gen_images.shape=}')
    print(f'{gen_images.dtype=}')

    m1 = torch.load(SCORE_DATASET_FID / config['dataset'] / 'm.pt', map_location=device)
    s1 = torch.load(SCORE_DATASET_FID / config['dataset'] / 's.pt', map_location=device)

    assert uncertainties.shape[0] == gen_images.shape[0], f"{uncertainties.shape=} {gen_images.shape=}"

    perc = args.perc  # Percentage of samples to remove

    i_sorted = torch.argsort(uncertainties)
    if args.mode == 'max':
        num_batches = math.ceil(len(uncertainties) * (1 - perc) // args.batch_size)
        mode_text = 'highest'
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        for i in tqdm(range(num_batches)):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            
            pred = predict_batch_fid_vector(model, gen_images_batch)

            pred_arr[i * args.batch_size: (i + 1) * args.batch_size] = pred
    elif args.mode == 'min':
        num_batches = len(uncertainties) // args.batch_size - math.ceil(len(uncertainties) * (perc) // args.batch_size)
        mode_text = 'lowest'
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        i_pred_arr = 0
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc) // args.batch_size), len(uncertainties) // args.batch_size)):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            pred = predict_batch_fid_vector(model, gen_images_batch)
            pred_arr[i_pred_arr * args.batch_size: (i_pred_arr + 1) * args.batch_size] = pred
            i_pred_arr += 1
    elif args.mode == 'both':
        num_batches = math.ceil(len(uncertainties) * (1 - (perc / 2)) // args.batch_size) - math.ceil(len(uncertainties) * (perc / 2) // args.batch_size)
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        i_pred_arr = 0
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc / 2) // args.batch_size), math.ceil(len(uncertainties) * (1 - (perc / 2)) // args.batch_size))):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = gen_images[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            pred = predict_batch_fid_vector(model, gen_images_batch)
            pred_arr[i_pred_arr * args.batch_size: (i_pred_arr + 1) * args.batch_size] = pred
        mode_text = 'outside the highest and lowest'
    
    m2 = pred_arr.mean(dim=0)
    s2 = torch.cov(pred_arr.T)

    fid_score_uncertainty = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID score with the top {}% samples with {} uncertainty removed: {:.3f}'.format(int(perc*100), mode_text, fid_score_uncertainty))

    i_random = torch.randperm(len(uncertainties))
    
    num_batches = math.ceil(len(uncertainties) * (1 - perc) // args.batch_size)
    pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
    for i in tqdm(range(math.ceil(len(uncertainties) * (1 - perc) // args.batch_size))):
        i_batch = i_random[i * args.batch_size: (i + 1) * args.batch_size]
        gen_images_batch = gen_images[i_batch]
        gen_images_batch = gen_images_batch.to(device)
        pred = predict_batch_fid_vector(model, gen_images_batch)
        pred_arr[i * args.batch_size : (i + 1) * args.batch_size] = pred

    m2_random = pred_arr.mean(dim=0)
    s2_random = torch.cov(pred_arr.T)

    fid_score_random = calculate_frechet_distance(m1, s1, m2_random, s2_random)

    return fid_score_uncertainty, fid_score_random

def compute_fid_score_bayesdiff_2(config, pred_arrays, uncertainties, args, device):
    dims = get_dims_bayesdiff(config['dataset'])

    # Add padding to gen_images
    # padding = nn.ZeroPad2d(((224-gen_images.shape[-1])//2, (224-gen_images.shape[-1])//2, (224-gen_images.shape[-2])//2, (224-gen_images.shape[-2])//2))
    # gen_images = padding(gen_images)



    m1 = torch.load(SCORE_DATASET_FID / config['dataset'] / 'm.pt', map_location=device)
    s1 = torch.load(SCORE_DATASET_FID / config['dataset'] / 's.pt', map_location=device)


    perc = args.perc  # Percentage of samples to remove

    i_sorted = torch.argsort(uncertainties)
    if args.mode == 'max':
        num_batches = math.ceil(len(uncertainties) * (1 - perc) // args.batch_size)
        mode_text = 'highest'
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        for i in tqdm(range(num_batches)):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = pred_arrays[i_batch]
            gen_images_batch = gen_images_batch.to(device)
            

            pred_arr[i * args.batch_size: (i + 1) * args.batch_size] = gen_images_batch
    elif args.mode == 'min':
        num_batches = len(uncertainties) // args.batch_size - math.ceil(len(uncertainties) * (perc) // args.batch_size)
        mode_text = 'lowest'
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        i_pred_arr = 0
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc) // args.batch_size), len(uncertainties) // args.batch_size)):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = pred_arrays[i_batch]
            gen_images_batch = gen_images_batch.to(device)

            pred_arr[i_pred_arr * args.batch_size: (i_pred_arr + 1) * args.batch_size] = gen_images_batch
            i_pred_arr += 1
    elif args.mode == 'both':
        num_batches = math.ceil(len(uncertainties) * (1 - (perc / 2)) // args.batch_size) - math.ceil(len(uncertainties) * (perc / 2) // args.batch_size)
        pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
        i_pred_arr = 0
        for i in tqdm(range(math.ceil(len(uncertainties) * (perc / 2) // args.batch_size), math.ceil(len(uncertainties) * (1 - (perc / 2)) // args.batch_size))):
            i_batch = i_sorted[i * args.batch_size: (i + 1) * args.batch_size]
            gen_images_batch = pred_arrays[i_batch]
            gen_images_batch = gen_images_batch.to(device)

            pred_arr[i_pred_arr * args.batch_size: (i_pred_arr + 1) * args.batch_size] = gen_images_batch
        mode_text = 'outside the highest and lowest'
    
    m2 = pred_arr.mean(dim=0)
    s2 = torch.cov(pred_arr.T)

    fid_score_uncertainty = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID score with the top {}% samples with {} uncertainty removed: {:.3f}'.format(int(perc*100), mode_text, fid_score_uncertainty))

    i_random = torch.randperm(len(uncertainties))
    
    num_batches = math.ceil(len(uncertainties) * (1 - perc) // args.batch_size)
    pred_arr = torch.zeros(num_batches * args.batch_size, dims, dtype=torch.float32)
    for i in tqdm(range(math.ceil(len(uncertainties) * (1 - perc) // args.batch_size))):
        i_batch = i_random[i * args.batch_size: (i + 1) * args.batch_size]
        gen_images_batch = pred_arrays[i_batch]
        gen_images_batch = gen_images_batch.to(device)

        pred_arr[i * args.batch_size : (i + 1) * args.batch_size] = gen_images_batch

    m2_random = pred_arr.mean(dim=0)
    s2_random = torch.cov(pred_arr.T)

    fid_score_random = calculate_frechet_distance(m1, s1, m2_random, s2_random)

    return fid_score_uncertainty, fid_score_random

@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('dataset_folders', type=str, nargs='+',  
        help='path to the dataset folder containing the uncertainty and gen_images files')

    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--perc', type=float, default=0.15, help='percentage of samples to remove with highest uncertainty')
    argparser.add_argument('--mode', type=str, default='max', choices=['max', 'min', 'both'], help='whether to remove samples with highest or lowest uncertainty')
    argparser.add_argument('--start-index', type=int, default=0, help='index of the first timestep to compute the total uncertainty')
    argparser.add_argument('--end-index', type=int, default=None, help='index of the last timestep to compute the total uncertainty')
    argparser.add_argument('--no-uncertainty-half', action='store_true', dest='no_uncertainty_half', help='whether to use the uncertainty half instead of float')
    argparser.add_argument('--fid-metric', type=str, default='bayesdiff', choices=['torchmetrics', 'bayesdiff'], help='which metric to use to compute the FID score')


    args = argparser.parse_args()

    # get canonical paths
    dataset_folders = resolve_dataset_folders_path(args.dataset_folders)
    args.dataset_folders = dataset_folders
                

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')
    assert_validity_folders_(args)

    model = load_inception_model_bayesdiff(device=device, dims=2048)
    model.eval()

    with open(Path(args.dataset_folders[0]) / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)

    scheduler = init_scheduler(config['dataset'])
    ddim_scheduler = DDIMScheduler.from_config(scheduler.config)
    ddim_scheduler.set_timesteps(config['generation_steps'])
    betas = ddim_scheduler.alphas_cumprod[ddim_scheduler.timesteps[args.start_index:args.end_index]]
    uncertainties = []
    for dataset_folder in args.dataset_folders:
        dataset_folder = Path(dataset_folder)
        print(f'Processing dataset folder {dataset_folder}')
    
        uncertainty_files = dataset_folder.files('uncertainty*.pth')
        assert len(uncertainty_files) > 0, f"no uncertainty files found in {dataset_folder}"
        gen_images_files = dataset_folder.files('gen_images*.pth')
        assert len(gen_images_files) > 0, f"no gen_images files found in {dataset_folder}"

        print('Loading uncertainty files...')
        for uncertainty_file in uncertainty_files:
            uncertainty = torch.load(uncertainty_file).cpu()
            if not args.no_uncertainty_half:
                uncertainty = uncertainty.half()
            print(f'{uncertainty.shape=}')
            uncertainty = uncertainty[:, args.start_index:args.end_index]
            # if uncertainty.shape[2] >= 3:
            #     uncertainty = uncertainty.amin(dim=2, keepdim=True)
            # uncertainty = (uncertainty - uncertainty.mean(dim=(1), keepdim=True)) / uncertainty.std(dim=(-1, -2, 1), keepdim=True)
            betas = betas.view(1, -1, 1, 1, 1)
            uncertainty = uncertainty 


            # uncertainty_min = uncertainty.amin(dim=1, keepdim=True)
            # uncertainty_max = uncertainty.amax(dim=1, keepdim=True)
            # uncertainty = (uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min)

            # uncertainty = uncertainty.square().mean(dim=(2, 3, 4), keepdim=True).sqrt()
            # uncertainty = uncertainty.sigmoid()

            # uncertainty = uncertainty.sqrt()

            # uncertainty = uncertainty.amax(dim=1, keepdim=True)

            uncertainty = uncertainty.sum(dim=(1, 2, 3, 4))
            uncertainties.append(uncertainty)
    uncertainties = torch.cat(uncertainties, dim=0)

    print('Done loading uncertainty files')
    print('Loading gen_images files...')

    pred_vectors = []
    for dataset_folder in args.dataset_folders:
        for gen_images_file in dataset_folder.files('gen_images*.pth'):
            gen_image = torch.load(gen_images_file).cpu()
            print(f'{gen_image.shape=}')
            for i in range(gen_image.shape[0]//args.batch_size + 1):
                batch_gen_image = gen_image[i*args.batch_size: (i+1)*args.batch_size]
                batch_gen_image = batch_gen_image.to(device)
                if args.fid_metric == 'torchmetrics':
                    pred = batch_gen_image * 255.
                    pred = pred.round().clamp(0, 255)
                    pred = pred.byte()
                else:
                    pred = predict_batch_fid_vector(model, batch_gen_image)

                pred_vectors.append(pred)
    pred_vectors = torch.cat(pred_vectors, dim=0)

    print('Statistics of gen_images:')
    print(f'Minimum: {batch_gen_image.min().item()}')
    print(f'Maximum: {batch_gen_image.max().item()}')
    if batch_gen_image.dtype == torch.float:
        print(f'Mean: {batch_gen_image.mean().item()}')
        print(f'Standard Deviation: {batch_gen_image.std().item()}')

    # print('Done loading gen_images files')

    perc = args.perc  # Percentage of samples to remove
    if args.fid_metric == 'torchmetrics':
        fid_score_uncertainty, fid_score_random = compute_fid_torchmetrics(config, pred_vectors, uncertainties, args, device)
    else:
        fid_score_uncertainty, fid_score_random = compute_fid_score_bayesdiff_2(config, pred_vectors, uncertainties, args, device)

    print(f"fid score without the top {int(perc*100)}% samples with random selection: {fid_score_random:.3f}")
    # Calculate fid scores
    fid_score = {
        "with_uncertainty": fid_score_uncertainty,
        "with_random_selection": fid_score_random,
        "mode": args.mode,
        "percentage": args.perc,
        "dataset": config['dataset'],
        "scheduler_type": config['scheduler_type'],
        "delta_fid": fid_score_random - fid_score_uncertainty,
        "paths": args.dataset_folders,
        "dropout": config['dropout'],
        "generation_steps": config['generation_steps'],
        "M": config['M'],
        "fid_metric": args.fid_metric,
    }

    # Save fid scores to JSON file
    json_file = RESULTS / "fid_scores.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            fid_scores = json.load(f)
        fid_scores.append(fid_score)
    else:
        fid_scores = [fid_score]

    with open(json_file, "w") as f:
        json.dump(fid_scores, f, indent=4)

    print(f"Fid scores saved in {json_file}")

def assert_validity_folders_(args):
    scheduler_type = None
    dataset = None
    for dataset_folder in args.dataset_folders:
        dataset_folder = Path(dataset_folder)
        assert dataset_folder.exists(), f"dataset folder {dataset_folder} does not exist"
        assert dataset_folder.joinpath('args.yaml').exists(), f"args.yaml not found in {dataset_folder}"
        with open(dataset_folder.joinpath('args.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        if dataset is None:
            dataset = config['dataset']
        else:
            assert dataset == config['dataset'], f"datasets do not match: {dataset=} {config['dataset']=}"
        if scheduler_type is None:
            scheduler_type = config['scheduler_type']
        else:
            assert scheduler_type == config['scheduler_type'], f"scheduler types do not match: {scheduler_type=} {config['scheduler_type']=}"
        assert dataset_folder.files('gen_images*.pth'), f"no gen_images files found in {dataset_folder}"
        assert dataset_folder.files('uncertainty*.pth'), f"no uncertainty files found in {dataset_folder}"



if __name__ == '__main__':
    main()