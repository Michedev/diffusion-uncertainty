import argparse
import numpy as np
from path import Path

import torch
from diffusion_uncertainty.metrics.precision_recall import IPR
from diffusion_uncertainty.paths import CONFIG, PR_MANIFOLD
from diffusion_uncertainty.utils import load_config

@torch.no_grad()
def main():
    args = parse_args()

    img_paths = [str(x.absolute()) for x in Path(args.path).walkfiles('*.png')]
    
    ipr = IPR(batch_size=args.batch_size, num_samples=args.num_samples, k=args.k)
    

    ipr.compute_manifold_ref(img_paths)

    manifold_ref = ipr.manifold_ref
    features, radii = manifold_ref.features, manifold_ref.radii

    if isinstance(manifold_ref.features, np.ndarray):
        features = torch.from_numpy(features)
    if isinstance(manifold_ref.radii, np.ndarray):
        radii = torch.from_numpy(radii)

    if not (PR_MANIFOLD / f'{args.dataset}').exists():
        (PR_MANIFOLD / f'{args.dataset}').mkdir()

    torch.save(features, PR_MANIFOLD / f'{args.dataset}' / 'features.pth')
    torch.save(radii, PR_MANIFOLD / f'{args.dataset}' / 'radii.pth')

    print(f'Saved features to {PR_MANIFOLD / f"{args.dataset}"}  ')
    print(f'Saved radii to {PR_MANIFOLD / f"{args.dataset}"} / radii.pth')

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--path', type=str, help='Path to the dataset folder')
    argparser.add_argument('--dataset', '--dataset-name', type=str, help='Dataset name')
    argparser.add_argument('--batch-size', type=int, default=32, help='Batch size', dest='batch_size')
    argparser.add_argument('--num-samples', type=int, default=10_000, help='Number of samples to use for the manifold reference', dest='num_samples')
    argparser.add_argument('-k', type=int, default=10, help='Value of k', dest='k')

    argparser.add_argument('--config', type=str, help='Path to the configuration file')

    args = argparser.parse_args()

    if args.config:
        print('Loading config file - ignoring other arguments')
        args = load_config(CONFIG / 'precision_recall_real', args.config)

    if args.path.startswith('~'):
        args.path = Path(args.path).expanduser()

    print(args)
    
    return args



if __name__ == '__main__':
    main()