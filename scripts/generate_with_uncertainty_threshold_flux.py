import argparse
import os
import torch
from path import Path

import pytorch_lightning as pl
from torchvision.utils import save_image
import yaml

from diffusion_uncertainty.paths import CONFIG, FLUX_UNCERTAINTY_GUIDANCE
from diffusion_uncertainty.pipeline_uncertainty.pipeline_flux_uncertainty import FluxPipelineUncertainty
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusion_uncertainty.utils import load_config
from diffusion_uncertainty import uncertainty_guidance

def main():
    args = parse_args()

    flux_path = 'black-forest-labs/FLUX.1-schnell'

    uncertainty_guidance.use_posterior = args.use_posterior

    seed = args.seed

    pl.seed_everything(seed)

    pipeline_uc: FluxPipelineUncertainty = FluxPipelineUncertainty.from_pretrained(pretrained_model_name_or_path=flux_path, torch_dtype=torch.bfloat16)     # type: ignore

    pipeline_uc = pipeline_uc.to('cuda')

    assert isinstance(pipeline_uc, FluxPipelineUncertainty), f'Expected StableDiffusion3PipelineUncertainty, got {type(pipeline_uc)}'

    output_sd_uc = pipeline_uc(
        prompt=args.prompt,
        num_inference_steps=args.num_steps, 
        generator=torch.Generator('cuda').manual_seed(seed),
        start_step_uc=args.start_step_threshold,
        num_steps_uc=args.num_steps_threshold,
        num_images_per_prompt=1,
        output_type='pt',
        return_dict=True,
        max_sequence_length=256,
        percentile=args.percentile,  
        lr=args.strength,      
    )
    i = 0
    while (FLUX_UNCERTAINTY_GUIDANCE / f'{i}').exists():
        i += 1
    dest_folder = FLUX_UNCERTAINTY_GUIDANCE / f'{i}'
    dest_folder.mkdir()

    with open(dest_folder / 'args.yaml', 'w') as f:
        yaml.safe_dump(args.__dict__, f)
    
    # print(pil_image)

    save_image(output_sd_uc['images'], dest_folder / 'output_flux_uc.png')

    # pil_image.save('output.png')
    del pipeline_uc
    if not args.skip_original:
        pipeline = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)     # type: ignore
        pipeline = pipeline.to('cuda')

        assert isinstance(pipeline, FluxPipeline), f'Expected StableDiffusionPipeline, got {type(pipeline)}'

        output = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_steps, 
            generator=torch.Generator('cuda').manual_seed(seed),
            num_images_per_prompt=1,
            output_type='pt',
            max_sequence_length=256,
            return_dict=True,
        )

        save_image(output['images'], dest_folder / 'output_flux.png')
    print(f'Saved to {dest_folder}')

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num-steps', type=int, default=20, help='number of steps to generate')
    argparser.add_argument('--prompt', type=str, default='a photo of a cat', help='prompt for the model')
    argparser.add_argument('--prompt-negative', default='', type=str, help='negative prompt for the model')
    # argparser.add_argument('--M', type=int, default=100, help='number of samples for the model')
    argparser.add_argument('--seed', type=int, default=491, help='seed for the model')
    argparser.add_argument('--start-step-threshold', type=int, default=0, help='step to start estimating the threshold')
    argparser.add_argument('--num-steps-threshold', type=int, default=20, help='number of steps to estimate the threshold')
    argparser.add_argument('--percentile', '--perc', '-p', type=float, default=0.95, help='percentile for the threshold', dest='percentile')    
    argparser.add_argument('--skip-original', action='store_true', help='skip original')
    argparser.add_argument('--use-posterior', action='store_true', help='use posterior', dest='use_posterior')
    argparser.add_argument('--strength', type=float, default=0.99, help='strength of the uncertainty guidance', dest='strength')


    argparser.add_argument('--config', type=str, help='Path to the configuration file')

    args: argparse.Namespace = argparser.parse_args()

    if args.config:
        print('Loading config file - ignoring other arguments')
        args_config = load_config(CONFIG / 'stable_diffusion_guidance', args.config)
        argparser.set_defaults(**args_config.__dict__)
        args = argparser.parse_args()

    return args


    


if __name__ == '__main__':
    main()