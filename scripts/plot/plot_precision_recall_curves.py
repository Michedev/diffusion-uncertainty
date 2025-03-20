from sympy import reduced_totient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math

import sys
from path import Path
from tqdm import tqdm
import yaml
from diffusion_uncertainty.metrics.precision_recall import IPR, Manifold
import matplotlib.pyplot as plt

from diffusion_uncertainty.paths import PR_MANIFOLD, RESULTS, SCORE_DATASET_FID
from scripts.compute_dataset_fid import calculate_frechet_distance
sys.path.append(Path(__file__).absolute().parent.parent.parent)

import argparse
import json
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import functional as F
from diffusion_uncertainty.paths import PRECISION_RECALL_CURVES



def plot_curves_precision_recall(uncertainties, pred_images, group_size, device, dataset_name, batch_size, num_groups):
    indexes = torch.linspace(0, len(uncertainties) - group_size, num_groups)
    indexes = [int(index.item()) for index in indexes]
    i_sorted_uncertainty = torch.argsort(uncertainties)

    groups = [pred_images[i_sorted_uncertainty[i:i+group_size]] for i in indexes]
    groups = [group.to(device) for group in groups]
    
    ipr = IPR(batch_size=batch_size)

    ipr.manifold_ref = Manifold(torch.load(PR_MANIFOLD / dataset_name / 'features.pth'), torch.load(PR_MANIFOLD / dataset_name / 'radii.pth'))

    print(ipr.manifold_ref)

    precision_recall_groups = [ipr.precision_and_recall(group.float()/255.) for group in groups]

    percentiles = [int(i * 100 / len(uncertainties)) for i in indexes]
    percentiles = [f'{percentile}th' for percentile in percentiles]

    fig, ax = plt.subplots()
    ax.plot(percentiles, [pr.precision for pr in precision_recall_groups], label='Precision')
    ax.plot(percentiles, [pr.recall for pr in precision_recall_groups], label='Recall')
    ax.set_xlabel('Percentile of uncertainty')

    fig.savefig(PRECISION_RECALL_CURVES / f'{dataset_name}.png')




@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('dataset_folders', type=str, nargs='+',  
        help='path to the dataset folder containing the uncertainty and gen_images files')

    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--start-index', type=int, default=0, help='index of the first timestep to compute the total uncertainty')
    argparser.add_argument('--end-index', type=int, default=-1, help='index of the last timestep to compute the total uncertainty')
    argparser.add_argument('--no-uncertainty-half', action='store_true', dest='no_uncertainty_half', help='whether to use the uncertainty half instead of float')
    argparser.add_argument('--num-groups', type=int, default=5, dest='num_groups', help='number of groups to divide the dataset into')
    argparser.add_argument('--group-size', type=int, default=100, dest='group_size', help='number of samples in each group')

    args = argparser.parse_args()

    # get canonical paths
    dataset_folders = []
    for dataset_folder in args.dataset_folders:
        dataset_folder = Path(dataset_folder)
        if not dataset_folder.exists():
            dataset_folder_2 = RESULTS / 'score-uncertainty' / dataset_folder
            if not dataset_folder_2.exists():
                raise ValueError(f"dataset folder {dataset_folder} does not exist")
            else:
                dataset_folders.append(dataset_folder_2)
        else:
            dataset_folders.append(dataset_folder)
    args.dataset_folders = dataset_folders
                

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')
    assert_validity_folders_(args)

    with open(Path(args.dataset_folders[0]) / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config['dataset']

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
            if uncertainty.shape[2] == 3:
                uncertainty = uncertainty.mean(dim=2, keepdim=True)
            uncertainty = (uncertainty - uncertainty.mean(dim=(-1, -2, 1), keepdim=True)) / uncertainty.std(dim=(-1, -2, 1), keepdim=True)

            uncertainty = uncertainty.sqrt()

            # uncertainty = uncertainty.amax(dim=1, keepdim=True)

            uncertainty = uncertainty.sum(dim=(1, 2, 3, 4))
            uncertainties.append(uncertainty)
    uncertainties = torch.cat(uncertainties, dim=0)

    print('Done loading uncertainty files')
    print('Loading gen_images files...')

    pred_images = []
    for dataset_folder in args.dataset_folders:
        for gen_images_file in dataset_folder.files('gen_images*.pth'):
            gen_image = torch.load(gen_images_file).cpu()
            print(f'{gen_image.shape=}')
            for i in range(gen_image.shape[0]//args.batch_size + 1):
                batch_gen_image = gen_image[i*args.batch_size: (i+1)*args.batch_size]
                batch_gen_image = batch_gen_image.to(device)

                pred_images.append(batch_gen_image)
    pred_images = torch.cat(pred_images, dim=0)


    plot_curves_precision_recall(uncertainties=uncertainties, pred_images=pred_images, group_size=args.group_size, device=device, dataset_name=dataset_name, batch_size=args.batch_size, num_groups=args.num_groups)


    
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