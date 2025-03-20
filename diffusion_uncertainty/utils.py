import argparse
from omegaconf import OmegaConf
from path import Path
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import adaptive_avg_pool2d
import yaml

from diffusion_uncertainty.init_model import init_model_scheduler, init_scheduler
from diffusion_uncertainty.paths import CONFIG, DIFFUSION_STARTING_POINTS, RESULTS
from diffusion_uncertainty.uvit.uvit import UViT
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.unets import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

MODEL_ID_DATASET = {
    'google/ddpm-ema-church-256': 'lsun-churches',
    'google/ddpm-ema-bed-256': 'lsun-beds',
    'google/ddpm-ema-cat-256': 'lsun-cats',
    'google/ddpm-ema-celebahq-256': 'celeba',
}

DATASET_IMAGE_SIZE = {
    'imagenet64': 64,
    'imagenet128': 128,
    'imagenet256': 256,
    'lsun-churches': 256,
    'imagenet512': 512,
    'cifar10': 32,
}

def plot_uncertainty_curve(uncertainty: torch.Tensor):
    mean_uncertainty = uncertainty.sum(dim=(-1, -2, -3), keepdim=False).mean(dim=0, keepdim=False).flatten()
    plt.plot(mean_uncertainty.flatten())
    std_uncertainty = uncertainty.sum(dim=(-1, -2, -3)).std(dim=(0), keepdim=False)
    print(mean_uncertainty)
    print(std_uncertainty)
    plt.fill_between(range(len(mean_uncertainty)), mean_uncertainty - std_uncertainty, mean_uncertainty + std_uncertainty, alpha=0.3)
    plt.show()


def load_config(folder_path: Path, config_name: str):
    """
    Load a configuration file from the specified folder path.

    Args:
        folder_path (Path): The path to the folder containing the configuration file.
        name (str): The name of the configuration file.

    Returns:
        config: The loaded configuration object.

    """
    if config_name.startswith('configs/'):
        config_name = Path(config_name).basename()
    if not config_name.endswith('.yaml'):
        config_name = config_name + '.yaml'
    with open(folder_path / config_name) as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    
    return config


def get_generation_configs(folder_name: str):
    configs = CONFIG.joinpath(folder_name).files('*.yaml')
    configs = [x.basename().replace('.yaml', '') for x in configs]
    return configs


def predict_batch_fid_vector(model, gen_images_batch):
    with torch.no_grad():
        *_, w, h = gen_images_batch.shape
        if gen_images_batch.dtype == torch.uint8:
            gen_images_batch = gen_images_batch.float() / 255.0
        gen_images_batch = (gen_images_batch - gen_images_batch.amin()) / (gen_images_batch.amax() - gen_images_batch.amin())

        pred = model(gen_images_batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu()

    return pred


def resolve_dataset_folders_path(dataset_folders: list[str]) -> list[Path]:
    """
    Resolves the paths of dataset folders. If a dataset folder path does not exist, it is searched under 'results/score-uncertainty'. If it is not found there, an error is raised.

    Args:
        dataset_folders (list[str]): A list of dataset folder paths.

    Returns:
        list[Path]: A list of `Path` objects representing the resolved dataset folder paths.

    Raises:
        ValueError: If a dataset folder path does not exist and no alternative path is found.

    """
    dataset_folders_output = []
    for dataset_folder in dataset_folders:
        dataset_folder = Path(dataset_folder)
        if not dataset_folder.exists():
            dataset_folder_2 = RESULTS / 'score-uncertainty' / dataset_folder
            if not dataset_folder_2.exists():
                raise ValueError(f"dataset folder {dataset_folder} does not exist")
            else:
                dataset_folders_output.append(dataset_folder_2)
        else:
            dataset_folders_output.append(dataset_folder)
    return dataset_folders_output



def load_X_T(dataset_name: str, start_index: int, num_samples: int) -> torch.Tensor:    
    return torch.load(DIFFUSION_STARTING_POINTS / dataset_name / 'X_T.pth')[start_index:start_index+num_samples]

def load_y(dataset_name: str, start_index: int, num_samples: int) -> torch.Tensor:
    return torch.load(DIFFUSION_STARTING_POINTS / dataset_name / 'y.pth')[start_index:start_index+num_samples]

def load_starting_points(dataset_name: str, start_index: int, num_samples: int):
    X_T = load_X_T(dataset_name, start_index, num_samples)
    y = load_y(dataset_name, start_index, num_samples)
    return X_T, y

def search_uncertainty_run_by(dataset_name: str, start_index: int, num_samples: int):
    runs = []
    for run in (RESULTS / 'score-uncertainty').dirs():
        if not run.joinpath('args.yaml').exists():
            continue
        with open(run / 'args.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if config['dataset'] != dataset_name:
            continue
        if config['start_index'] != start_index:
            continue
        if config['num_samples'] < num_samples:
            continue
        runs.append(run)
    return runs

def get_betas_from_run(run: Path):
    with open(run / 'args.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset = config['dataset']
    return get_betas_from_dataset_name(dataset)

def get_betas_from_dataset_name(dataset_name: str, num_timesteps: int):
    scheduler = init_scheduler(dataset_name)
    ddim_scheduler = DDIMScheduler.from_config(scheduler.config)
    ddim_scheduler.set_timesteps(num_timesteps)
    return scheduler.betas[ddim_scheduler.timesteps]

def predict_model(model: UNet2DModel | UViT | UViTAE | torch.nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
    if isinstance(t, float):
        t = round(t)
    if isinstance(t, int):
        t = torch.zeros(size=(x.shape[0],), dtype=torch.int64, device=x.device).fill_(t)
    if isinstance(model, (UViT, UViTAE)):
        return model(x, t, y)
    elif isinstance(model, UNet2DModel):
        return model(x, t).sample
    else:
        return model(x, t, y=y)[:, :3]
    
def load_y(args: argparse.Namespace | str) -> torch.Tensor:
    if isinstance(args, argparse.Namespace):
        if hasattr(args, 'dataset'):
            dataset = args.dataset
        elif hasattr(args, 'dataset_name'):
            dataset = args.dataset_name
        else:
            raise ValueError(f"args must have 'dataset' or 'dataset_name' attribute, got {args}")
        return torch.load(DIFFUSION_STARTING_POINTS / dataset / 'y.pth', map_location='cpu')
    elif isinstance(args, str):
        return torch.load(DIFFUSION_STARTING_POINTS / args / 'y.pth', map_location='cpu')
    else:
        raise ValueError(f"args must be either Namespace or str, got {type(args)}")

