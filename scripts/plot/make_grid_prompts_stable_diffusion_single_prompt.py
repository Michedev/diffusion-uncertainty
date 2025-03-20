import argparse
from collections import defaultdict
from beartype import beartype
import numpy as np
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import yaml
from diffusion_uncertainty.pipeline_uncertainty.pipeline_stable_diffusion_uncertainty_guided import StableDiffusionPipelineUncertainty
from diffusion_uncertainty.paths import RESULTS, STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE
from torchvision.io import read_image
from torchvision.utils import save_image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import subprocess

def search_existing_image(prompt, percentile, seed, start_step_threshold, num_steps_threshold, num_steps) -> torch.Tensor | None:
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
        if folder.basename() == 'grids':
            continue
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

def search_original_image(prompt, seed, num_steps):
    variables = {
        'prompt': prompt,
        'seed': seed,
        'num_steps': num_steps,
    }
    for folder in STABLE_DIFFUSION_UNCERTAINTY_GUIDANCE.dirs():
        if folder.basename() == 'grids':
            continue
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
        if same_config and (folder / 'output_sd.png').exists():
            return read_image(folder / 'output_sd.png')
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

    argparser = argparse.ArgumentParser()

    argparser.add_argument('prompt', type=str, help='Stable Diffusion prompts')
    argparser.add_argument('--start-step-threshold', type=int, default=0, help='Start step for threshold')
    argparser.add_argument('--seed', type=int, default=491, help='Seed for the stable diffusion')

    args = argparser.parse_args()

    print('args:')
    print(args)

    new_folder = get_new_folder_stable_diffusion_uncertainty()

    i_folder = int(new_folder.basename())
    
    for i, percentile in enumerate([0.99, 0.97, 0.95, 0.92, 0.9]):
        if i != 0:
            suffix = ['--skip-original']
        else:
            suffix = []
        subprocess.run(['python', 'scripts/stable_diffusion_generate_with_uncertainty_threshold.py', '--prompt', f'"{args.prompt}"', '--percentile', str(percentile), '--start-step-threshold', str(args.start_step_threshold), '--num-steps', '20', '--seed', str(args.seed), '--num-steps-threshold', '3'] + suffix, check=True)

        if i != 0:
            new_folder.joinpath('output_sd.png').copy(new_folder.parent / f'{i_folder + i}' / 'output_sd.png')
    
    subprocess.run(['python', 'scripts/plot/assemble_row_sd.py'] + [str(i_folder + j) for j in range(5)], check=True)


                        

if __name__ == '__main__':
    main()