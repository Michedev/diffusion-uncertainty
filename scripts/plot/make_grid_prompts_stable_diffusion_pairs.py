import argparse
from collections import defaultdict
from beartype import beartype
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import yaml
from diffusion_uncertainty.pipeline_uncertainty.pipeline_stable_diffusion_uncertainty_guided import StableDiffusionPipelineUncertainty
from diffusion_uncertainty.paths import STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE
from torchvision.io import read_image
from torchvision.utils import save_image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from torchvision.utils import make_grid

def search_existing_image(prompt, percentile, seed, start_step_threshold, num_steps_threshold, num_steps):
    # Config example
    # num_steps: 20
    # num_steps_threshold: 4
    # percentile: 0.99
    # prompt: dog walking on the moon, realistic
    # prompt_negative: ''
    # seed: 4238
    # start_step_threshold: 0

    variables = {
        'prompt': prompt,
        'percentile': percentile,
        'seed': seed,
        'start_step_threshold': start_step_threshold,
        'num_steps_threshold': num_steps_threshold,
        'num_steps': num_steps,
    }
    for folder in STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE.dirs():
        with open(folder / 'args.yaml', 'r') as f:
            config = yaml.safe_load(f)
        same_config = True
        for k, v in variables.items():
            if k not in config:
                same_config = False
                break
            if config[k] != v:
                same_config = False
                break
        if same_config and (folder / 'output_sd_uc.png').exists():
            return read_image(folder / 'output_sd_uc.png')
    return None

@beartype
def image_for_plot(image: torch.Tensor) -> np.ndarray:
    assert len(image.shape) == 3 or (len(image.shape) == 4 and image.shape[0] == 1), f'Expected 3 dimensions, got {(image.shape)}'
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = image[0]
    if image.shape[0] in (1, 3):
        # channel first to channel last
        image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    return image


def get_new_folder_stable_diffusion_uncertainty():
    i = 0
    while (STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / f'{i}').exists():
        i += 1
    return STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / f'{i}'

@torch.no_grad()
def main():
    """
    This function processes stable diffusion folders and creates a grid of images.

    Args:
        folders (str): List of stable diffusion folders.

    Returns:
        None
    """

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('folders', type=str, nargs='+', help='Stable Diffusion folders')
    args = argparser.parse_args()

    uc_images = []
    images = []

    # Process each folder
    for folder in args.folders:
        print('Processing', folder)
        folder = STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / folder

        # Check if required files exist
        assert ((folder / 'args.yaml').exists())
        assert ((folder / 'output_sd_uc.png').exists())
        assert ((folder / 'output_sd.png').exists())

        # Read images
        img = read_image(folder / 'output_sd.png')
        uc_img = read_image(folder / 'output_sd_uc.png')

        # Normalize images
        img =  img / 255.0
        uc_img = uc_img / 255.0 
        
        images.append(img.unsqueeze(0))
        uc_images.append(uc_img.unsqueeze(0))

    # Create grid of images
    grid_images = torch.stack([torch.cat(images, dim=0), torch.cat(uc_images, dim=0)], dim=0).flatten(0, 1)
    print(grid_images.shape)
    grid_images = make_grid(grid_images, nrow=len(uc_images), pad_value=1.0, padding=20)

    # Save grid image
    grid_folder = STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE / 'grids'
    grid_folder.mkdir_p()
    i = 0
    while (grid_folder / f'{i}.png').exists():
        i += 1
    save_image(grid_images, grid_folder / f'{i}.png')

    print(f'Saved grid to {grid_folder / f"{i}.png"}')

            
                        

if __name__ == '__main__':
    main()