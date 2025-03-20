"""
Compute time-step wise threshold for each pixel of the diffusion process.
"""


from path import Path
import sys

from tqdm import tqdm

from diffusion_uncertainty.utils import load_config


sys.path.append(Path(__file__).absolute().parent.parent)

from diffusion_uncertainty.paths import CONFIG, FID, RESULTS, ROOT
import argparse
import yaml
import torch

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



def main():
    """
    Generate a threshold for each pixel in the image, for each timestep.
    It saves the thresholds in a file in RESULTS/thresholds/dataset/thresholds_{scheduler_type}.pth
    The saved tensor has shape (timesteps, 1, height, width)
    """

    args = parse_args()

    # get canonical paths
    dataset_folders = []
    for dataset_folder in args.dataset_folders:
        dataset_folder = Path(dataset_folder)
        if not dataset_folder.exists():
            dataset_folder = RESULTS / 'score-uncertainty' / dataset_folder
            if not dataset_folder.exists():
                raise ValueError(f"dataset folder {dataset_folder} does not exist")
            else:
                dataset_folders.append(dataset_folder)
        else:
            dataset_folders.append(dataset_folder)
    args.dataset_folders = dataset_folders

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')
    assert_validity_folders_(args)

    with open(Path(args.dataset_folders[0]) / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print('Config:')
    print(config)

    uncertainties = load_uncertainty_datasets(args.dataset_folders, args.perc)

    if len(uncertainties) == 1:
        timestep_thresholds = uncertainties[0].squeeze(0)
    else:
        uncertainties: torch.Tensor = torch.cat(uncertainties, dim=0)
        # if uncertainties.shape[2] == 3:
        #     uncertainties = uncertainties.amax(dim=2, keepdim=True)

        print('Done loading uncertainty files')

        num_samples, timesteps, _, height, width = uncertainties.shape
        timestep_thresholds = []


        for i in tqdm(range(timesteps)):
            uncertainties_timestep = uncertainties[:, i]
            i_uncertaintities = uncertainties_timestep.argsort(dim=0)

            # take 15th percentile
            i_perc_th = i_uncertaintities[int(num_samples * args.perc)].unsqueeze(0)

            uncertainty_perc_th = uncertainties_timestep.gather(dim=0, index=i_perc_th).squeeze(0)
                    
            timestep_thresholds.append(uncertainty_perc_th)
        timestep_thresholds = torch.stack(timestep_thresholds, dim=0)
    
    print(f'{timestep_thresholds.shape=}')
    threshold_folder = RESULTS / 'thresholds'
    if not threshold_folder.exists():
        threshold_folder.mkdir()
    threshold_folder = threshold_folder / config['dataset']
    if not threshold_folder.exists():
        threshold_folder.mkdir()
    torch.save(timestep_thresholds, threshold_folder / f'thresholds_{config["scheduler_type"]}_perc={args.perc}.pth')
    print('Saved thresholds to', threshold_folder / f'thresholds_{config["scheduler_type"]}_perc={args.perc}.pth')
    
    args_dict = args.__dict__.copy()
    args_dict['dataset_config'] = config
    args_dict['dataset_folders'] = [str(x.relpath(ROOT)) for x in args_dict['dataset_folders']]
    with open(threshold_folder / f'config_{config["scheduler_type"]}_perc={args.perc}.yaml', 'w') as f:
        yaml.safe_dump(args_dict, f, )

def load_uncertainty_datasets(dataset_folders, perc):
    """
    Load uncertainty datasets from the given dataset folders and compute the 15th percentile threshold for each timestep.

    Args:
        dataset_folders (list): A list of dataset folders containing uncertainty files.
        perc (float): The percentile value used to compute the threshold.

    Returns:
        list: A list of tensors representing the computed thresholds for each timestep.

    Raises:
        AssertionError: If no uncertainty files or gen_images files are found in any of the dataset folders.
    """
    uncertainties = []
    for dataset_folder in dataset_folders:
        dataset_folder = Path(dataset_folder)
        print(f'Processing dataset folder {dataset_folder}')
    
        uncertainty_files = dataset_folder.files('uncertainty_*.pth')
        assert len(uncertainty_files) > 0, f"no uncertainty files found in {dataset_folder}"
        gen_images_files = dataset_folder.files('gen_images_*.pth')
        assert len(gen_images_files) > 0, f"no gen_images files found in {dataset_folder}"

        print('Loading uncertainty files...')
        for uncertainty_file in uncertainty_files:
            uncertainty = torch.load(uncertainty_file).cpu().half()
            print(f'{uncertainty.shape=}')
            num_samples = uncertainty.shape[0]
            timesteps = uncertainty.shape[1]

            if uncertainty.shape[0] < 100:
                continue

            timestep_thresholds = []
            for i in tqdm(range(timesteps)):
                uncertainties_timestep = uncertainty[:, i]
                i_uncertaintities = uncertainties_timestep.argsort(dim=0)

                # take 15th percentile
                i_15th = i_uncertaintities[int(num_samples * perc)].unsqueeze(0)

                uncertainty_15th = uncertainties_timestep.gather(dim=0, index=i_15th).squeeze(0)
                        
                timestep_thresholds.append(uncertainty_15th)
            timestep_thresholds = torch.stack(timestep_thresholds, dim=0)
            uncertainties.append(timestep_thresholds.unsqueeze(0))
    return uncertainties

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset-folders', '--datasets', type=str, nargs='+', required=False, help='path to the dataset folder containing the uncertainty and gen_images files', dest='dataset_folders')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--perc', type=float, default=0.15, help='percentage of samples to estimate threshold from')

    options = CONFIG.joinpath('threshold').files('*.yaml')
    options = [x.basename().replace('.yaml', '') for x in options]

    argparser.add_argument('--config', type=str, help='path to the config file', choices=options)

    args = argparser.parse_args()

    if args.config is not None:
        print('Loading config file - ignoring other arguments')
        args = load_config(CONFIG.joinpath('threshold'), args.config)

    return args

if __name__ == '__main__':
    main()