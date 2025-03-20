import sys
from path import Path
import yaml

sys.path.append(Path(__file__).absolute().parent.parent)

import torch

from diffusion_uncertainty.metrics.nll import run_bpd_evaluation  
from diffusion_uncertainty.paths import BPD
import argparse
from datetime import datetime



def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset', '--dataset-name', type=str, default='imagenet64', help='The name of the dataset to use.')
    argparser.add_argument('--start-index', type=int, default=0, help='The starting index of the samples to evaluate.')
    argparser.add_argument('--num-samples', type=int, default=1000, help='The number of samples to evaluate.')
    argparser.add_argument('--batch-size', type=int, default=64, help='The batch size for evaluation.')

    args = argparser.parse_args()

    element_wise_bpd = run_bpd_evaluation(args.dataset, args.start_index, args.num_samples, args.batch_size)

    run_folder = BPD / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder.mkdir()

    with open(run_folder / 'args.yaml', 'w') as f:
        yaml.safe_dump(args.__dict__, f)
    print('Saved args to', run_folder / 'args.yaml')

    torch.save(element_wise_bpd, run_folder / 'bpd.pth')
    print('Saved element-wise BPD to', run_folder / 'bpd.pth')
    

if __name__ == '__main__':
    main()