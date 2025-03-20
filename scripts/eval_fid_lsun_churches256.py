"""
Compare FID Score obtained from the pretrained model with the finetuned model.
"""
import json
import sys
from path import Path

from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))


from diffusion_uncertainty.paths import DATASET_FID
from diffusion_uncertainty.fid import load_real_fid_model
import torchmetrics
import argparse
from diffusers.pipelines import DDIMPipeline, DDPMPipeline
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipImagenetClassConditioned

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--use-ddpm-sampler', '--use--ddpm', action='store_true', dest='use_ddpm_sampler')
    parser.add_argument('--num-samples', type=int, default=50000)
    parser.add_argument('--num-inference-steps', '--steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')

    if args.use_ddpm_sampler and args.num_inference_steps is None:
        args.num_inference_steps = 1000
    elif not args.use_ddpm_sampler and args.num_inference_steps is None:
        args.num_inference_steps = 50

    FID_LSUN_CHURCHES256 = DATASET_FID / 'lsun-churches256'

    lsun_fid = load_real_fid_model(FID_LSUN_CHURCHES256, device=device, normalize=True)

    model_id = "google/ddpm-ema-church-256"

    ddpm_pipeline = DDPMPipeline.from_pretrained(model_id) if args.use_ddpm_sampler else DDIMPipeline.from_pretrained(model_id)

    ddpm_pipeline.to(device)

    lsun_fid_score = generate_samples_and_compute_fid(args.num_samples, args.num_inference_steps, args.batch_size, lsun_fid, ddpm_pipeline, device=device)

    print(f"FID Score normal model: {lsun_fid_score}")


    print(f'Loading finetuned model from checkpoint located at {args.model_path}')

    state_dict = torch.load(args.model_path)['state_dict']
    state_dict = {k.replace('unet.', ''): v for k, v in state_dict.items() if k.startswith('unet.')}
    ddpm_pipeline.unet.load_state_dict(state_dict)

    lsun_fid = load_real_fid_model(FID_LSUN_CHURCHES256, device=device, normalize=True)

    fid_score_finetuned = generate_samples_and_compute_fid(args.num_samples, args.num_inference_steps, args.batch_size, lsun_fid, ddpm_pipeline, device=device)

    print(f"FID Score normal model: {lsun_fid_score}")

    print(f"FID Score finetuned model: {fid_score_finetuned}")

    print(f"Improvement: {lsun_fid_score - fid_score_finetuned}")

    results_dict = {'lsun_fid_score': lsun_fid_score, 'fid_score_finetuned': fid_score_finetuned, 'improvement': lsun_fid_score - fid_score_finetuned}
    dest_path = Path(args.model_path).parent / 'fid_scores.json'
    
    with open(dest_path.absolute(), 'w') as f:
        json.dump(results_dict, f, indent=4)


@torch.no_grad()
def generate_samples_and_compute_fid(num_samples, num_inference_steps, batch_size, fid_evaluator, pipeline, device=None):
    """
    Generates samples using the given pipeline and computes the Fréchet Inception Distance (FID) score.

    Args:
        num_samples (int): The total number of samples to generate.
        num_inference_steps (int): The number of inference steps to perform in the pipeline.
        batch_size (int): The batch size to use for generating samples.
        image_size (int): The size of the generated images (assumed to be square).
        fid_evaluator: The FID evaluator object used to compute the FID score.
        pipeline: The pipeline object used for generating samples.
        device: The device to use for generating samples (default: None).

    Returns:
        float: The computed FID score.
    """

    num_generated_samples = 0
    first = True
    while num_samples > num_generated_samples:
        tqdm.tqdm.write(f"Generated samples: {num_generated_samples} / {num_samples}")

        gen_images = pipeline(num_inference_steps=num_inference_steps, batch_size=batch_size, output_type='torch', return_dict=True, device=device).images

        # gen_images = (input / 2 + 0.5).clamp(0, 1)
        # gen_images = gen_images.cpu().numpy()

        if isinstance(gen_images, np.ndarray):
            gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
            if device is not None:
                gen_images = gen_images.to(device)
        num_generated_samples += gen_images.shape[0]

        if first:
            print(gen_images.shape)
            print('min:', gen_images.amin())
            print('max:', gen_images.amax())
            first = False

        gen_images = gen_images * 255.0
        gen_images = gen_images.round()
        gen_images = gen_images.to(torch.uint8)

        print(gen_images.shape)

        fid_evaluator.update(gen_images, real=False)

        del gen_images

    fid_score = fid_evaluator.compute()

    return fid_score




if __name__ == '__main__':
    main()
