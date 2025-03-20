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

from diffusion_uncertainty.paths import CONFIG, PR_MANIFOLD, RESULTS, SCORE_DATASET_FID
from diffusion_uncertainty.utils import load_config
from scripts.compute_dataset_fid import calculate_frechet_distance
sys.path.append(Path(__file__).absolute().parent.parent)

from diffusion_uncertainty.fid import get_dims_bayesdiff, load_inception_model_bayesdiff, load_real_fid_model
import argparse
import json
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn import functional as F



def compute_precision_recall(uncertainties, pred_images, perc, device, dataset_name, mode, batch_size, k):# -> tuple[Any, Any]:
    num_reduced_samples = int(len(uncertainties) * (1 - perc))
    ipr = IPR(num_samples=num_reduced_samples, batch_size=batch_size, k=k)

    features_ref = torch.load(PR_MANIFOLD / dataset_name / 'features.pth')
    radii_ref = torch.load(PR_MANIFOLD / dataset_name / 'radii.pth')

    manifold_ref = Manifold(features_ref, radii_ref)

    ipr.manifold_ref = manifold_ref

    i_sorted = torch.argsort(uncertainties)
    if mode == 'max':
        mode_text = 'highest'
        pr_uc = ipr.precision_and_recall(pred_images[i_sorted[:num_reduced_samples]].float()/255.)

    elif mode == 'min':
        mode_text = 'lowest'
        pr_uc = ipr.precision_and_recall(pred_images[i_sorted[-num_reduced_samples:]].float()/255.)

    print(f"Precision and recall with {mode_text} uncertainty:")
    print(f"Precision: {pr_uc.precision:.3f}")
    print(f"Recall: {pr_uc.recall:.3f}")

    i_random = torch.randperm(len(uncertainties))
    
    pr_random = ipr.precision_and_recall(pred_images[i_random[:num_reduced_samples]].float()/255.)

    print(f"Precision and recall with random selection:")
    print(f"Precision: {pr_random.precision:.3f}")
    print(f"Recall: {pr_random.recall:.3f}")

    return pr_random, pr_uc

@torch.no_grad()
def main():
    args = parse_args()

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

    uncertainties = load_uncertainties(args)

    print('Done loading uncertainty files')
    print('Loading gen_images files...')

    pred_images = load_generated_images(args, device)



    perc = args.perc  # Percentage of samples to remove
    random_pr, uc_pr = compute_precision_recall(uncertainties=uncertainties, pred_images=pred_images, perc=perc, device=device, dataset_name=dataset_name, mode=args.mode, batch_size=args.batch_size, k=args.k)


    # Calculate fid scores
    pr_score = {
        "random_precision": random_pr.precision,
        "random_recall": random_pr.recall,
        "uc_precision": uc_pr.precision,
        "uc_recall": uc_pr.recall,
        "mode": args.mode,
        "percentage": args.perc,
        "dataset": config['dataset'],
        "scheduler_type": config['scheduler_type'],
        "delta_precision": uc_pr.precision - random_pr.precision,
        "delta_recall": uc_pr.recall - random_pr.recall,
        "paths": args.dataset_folders,
        "dropout": config['dropout'],
        "generation_steps": config['generation_steps'],
        "M": config['M'],
        'k': args.k,
    }
    for k in pr_score:
        if isinstance(pr_score[k], torch.Tensor):
            pr_score[k] = pr_score[k].item()

    # Save fid scores to JSON file
    json_file = RESULTS / "pr_scores.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            fid_scores = json.load(f)
        fid_scores.append(pr_score)
    else:
        fid_scores = [pr_score]

    with open(json_file, "w") as f:
        json.dump(fid_scores, f, indent=4)

    print(f"Precision-recall scores saved in {json_file}")

def load_uncertainties(args):
    """
    Load uncertainties from dataset folders and process them.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        torch.Tensor: Concatenated uncertainties.

    Raises:
        AssertionError: If no uncertainty files or gen_images files are found in the dataset folder.

    """
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

            # uncertainty_min = uncertainty.amin(dim=1, keepdim=True)
            # uncertainty_max = uncertainty.amax(dim=1, keepdim=True)
            # uncertainty = (uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min)

            # uncertainty = uncertainty.square().mean(dim=(2, 3, 4), keepdim=True).sqrt()
            # uncertainty = uncertainty.sigmoid()

            uncertainty = uncertainty.sqrt()

            # uncertainty = uncertainty.amax(dim=1, keepdim=True)

            uncertainty = uncertainty.sum(dim=(1, 2, 3, 4))
            uncertainties.append(uncertainty)
    uncertainties = torch.cat(uncertainties, dim=0)
    return uncertainties

def load_generated_images(args, device):
    pred_images = []
    for dataset_folder in args.dataset_folders:
        for gen_images_file in dataset_folder.files('gen_images*.pth'):
            gen_image = torch.load(gen_images_file).cpu()
            print(f'{gen_image.shape=}')
            for i in range(gen_image.shape[0]//args.batch_size + 1):
                batch_gen_image = gen_image[i*args.batch_size: (i+1)*args.batch_size]
                batch_gen_image = batch_gen_image.to(device)
                # pred = predict_batch_fid_vector(model, batch_gen_image)

                pred_images.append(batch_gen_image)
    pred_images = torch.cat(pred_images, dim=0)
    return pred_images

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('dataset_folders', type=str, nargs='+', help='path to the dataset folder containing the uncertainty and gen_images files')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--perc', type=float, default=0.15, help='percentage of samples to remove with highest uncertainty')
    argparser.add_argument('--mode', type=str, default='max', choices=['max', 'min', 'both'], help='whether to remove samples with highest or lowest uncertainty')
    argparser.add_argument('--start-index', type=int, default=0, help='index of the first timestep to compute the total uncertainty')
    argparser.add_argument('--end-index', type=int, default=-1, help='index of the last timestep to compute the total uncertainty')
    argparser.add_argument('--no-uncertainty-half', action='store_true', dest='no_uncertainty_half', help='whether to use the uncertainty half instead of float')
    argparser.add_argument('--fid-metric', type=str, default='bayesdiff', choices=['torchmetrics', 'bayesdiff'], help='which metric to use to compute the FID score')
    argparser.add_argument('-k', default=3, dest='k', type=int)

    argparser.add_argument('--config', type=str, help='path to the config file')

    args = argparser.parse_args()

    if args.config:
        args = load_config(CONFIG / 'precision_recall', args.config)
        argparser.set_defaults(**args.__dict__)

        args = argparser.parse_args()

    return args


    
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