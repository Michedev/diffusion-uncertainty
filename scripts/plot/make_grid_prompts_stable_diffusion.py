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


def search_existing_image(prompt, percentile, seed, start_step_threshold, num_steps_threshold, num_steps):
    # Config example
    # num_steps: 20
    # num_steps_threshold: 4
    # percentile: 0.99
    # prompt: dog walking on the moon, realistic
    # prompt_negative: ''
    # seed: 4238
    # start_step_threshold: 0

    variables = locals()
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

    argparser = argparse.ArgumentParser()

    argparser.add_argument('prompts', type=str, nargs='+', help='Stable Diffusion prompts')
    argparser.add_argument('--percentiles', type=float, nargs='+', help='Thresholds for the prompts', default=[0.99, 0.98, 0.97, 0.96, 0.95])
    argparser.add_argument('--seed', type=int, default=491, help='seed for the model')
    argparser.add_argument('--start-step-threshold', type=int, default=0, help='step to start estimating the threshold')
    argparser.add_argument('--num-steps-threshold', type=int, default=20, help='number of steps to estimate the threshold')
    argparser.add_argument('--num-steps', type=int, default=20, help='number of steps to generate')


    args = argparser.parse_args()



    pipeline_uc: StableDiffusionPipelineUncertainty = StableDiffusionPipelineUncertainty.from_pretrained('runwayml/stable-diffusion-v1-5')     # type: ignore
    pipeline_uc = pipeline_uc.to('cuda')

    fig, axs = plt.subplots(len(args.prompts), len(args.percentiles) + 1, figsize=(5 * len(args.percentiles), 5 * (len(args.prompts)+1))) # +1 for the the original image
    uc_images_to_save = defaultdict(list)
    config_to_save = defaultdict(list)
    for i, prompt in enumerate(args.prompts):
        for j, percentile in enumerate(args.percentiles):
            pl.seed_everything(args.seed)
            j1 = j + 1
            image = search_existing_image(prompt, percentile, args.seed, args.start_step_threshold, args.num_steps_threshold, args.num_steps)
            if image is None:
                image = pipeline_uc(
                    prompt=prompt,
                    num_inference_steps=args.num_steps, 
                    generator=torch.Generator('cuda').manual_seed(args.seed),
                    start_step_uc=args.start_step_threshold,
                    num_steps_uc=args.num_steps_threshold,
                    num_images_per_prompt=1,
                    output_type='pt',
                    return_dict=True,
                    percentile=percentile,
                )['images'] # type: ignore
                uc_images_to_save[prompt].append(image.cpu())
                config_to_save[prompt].append({
                    'prompt': prompt,
                    'percentile': percentile,
                    'seed': args.seed,
                    'start_step_threshold': args.start_step_threshold,
                    'num_steps_threshold': args.num_steps_threshold,
                    'num_steps': args.num_steps,
                })

            image = image_for_plot(image)

            axs[i, j1].imshow(image)

            axs[i, j1].set_xticks([])
            axs[i, j1].set_yticks([])

    del pipeline_uc

    pipeline = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')     # type: ignore

    for i, prompt in enumerate(args.prompts):
        pl.seed_everything(args.seed)
        image = pipeline(prompt=prompt, num_inference_steps=args.num_steps, num_images_per_prompt=1, output_type='pt', return_dict=True)['images']
        image: np.ndarray = image_for_plot(image)
        axs[i, 0].imshow(image)
        for uc_image_to_save, config in zip(uc_images_to_save[prompt], config_to_save[prompt]):
            dest_folder = get_new_folder_stable_diffusion_uncertainty()
            dest_folder.mkdir()
            save_image(uc_image_to_save, dest_folder / 'output_sd_uc.png')
            with open(dest_folder / 'args.yaml', 'w') as f:
                yaml.safe_dump(config, f)
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1)
            save_image(image, dest_folder / 'output_sd.png')


    for i, prompt in enumerate(args.prompts):
        # axs[i, 0].set_title(prompt)
        axs[i, 0].set_ylabel(prompt)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

    for j, percentile in enumerate(args.percentiles):
        axs[-1, j+1].set_xlabel(f'{int(percentile*100)}%')
    axs[-1, 0].set_xlabel('Original')

    fig.tight_layout()
    fig.savefig('grid.png')

            
                        

if __name__ == '__main__':
    main()