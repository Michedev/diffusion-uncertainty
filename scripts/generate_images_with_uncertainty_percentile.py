import argparse
import torch
from path import Path
import yaml
import pytorch_lightning as pl
from diffusion_uncertainty.fid import load_real_fid_model
from diffusion_uncertainty.generate_samples import generate_samples_model_scheduler_class_conditioned, generate_samples_model_scheduler_class_conditioned_with_percentile
from diffusion_uncertainty.init_model import init_guided_diffusion_imagenet128_and_scheduler, init_guided_diffusion_imagenet64_and_scheduler
from diffusion_uncertainty.paths import CONFIG, FID, RESULTS, ROOT, THRESHOLD
from diffusion_uncertainty.pipeline_uncertainty.uncertainty_guidance import generate_samples_model_scheduler_class_conditioned_with_threshold
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlip
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertainty
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_single import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintySingle
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip_grad import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipGrad
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_image import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyImage
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_single_score import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintySingleScore
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered_d import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCenteredD
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_zigzag_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyZigZagCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyOriginal
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout
from diffusion_uncertainty.utils import load_config
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

dataset_model_map = {
    'imagenet64': ROOT.joinpath('models/64x64_diffusion.pth'),
    'imagenet128': ROOT.joinpath('models/128x128_diffusion.pth'),
}

def instantiate_uc_scheduler(args, scheduler, unet, y):
    if args.scheduler_type == 'uncertainty':
        uc_scheduler = DDIMSchedulerUncertaintyUncertainty.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_image':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyImage.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'flip':
        uc_scheduler = DDIMSchedulerUncertaintyFlip.from_config(scheduler.config, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, eta=args.eta, prompt_embeds=y)
    elif args.scheduler_type == 'flip_grad':
        uc_scheduler = DDIMSchedulerUncertaintyFlipGrad.from_config(scheduler.config, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, eta=args.eta, prompt_embeds=y)
    elif args.scheduler_type == 'uncertainty_single':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintySingle.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_single_score':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintySingleScore.from_config(scheduler.config, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_centered':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyCentered.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_original':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyOriginal.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=False, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_centered_d':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyCenteredD.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, y=y, eta=args.eta, uncertainty_distance=args.uncertainty_distance)
    elif args.scheduler_type == 'uncertainty_zigzag_centered':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyZigZagCentered.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, y=y, eta=args.eta,
        num_zigzag=args.num_zigzag)
    else:
        uc_scheduler = DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout.from_config(scheduler.config, prompt_embeds=y, unet=unet, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, eta=args.eta, dropout=args.dropout)

    return uc_scheduler

@torch.no_grad()
def main():
    args = parse_args()

    args.eta = 0.0
    args.M = 5
    args.scheduler_type = 'uncertainty'
    args.predict_next = False
    args.start_step_uc = args.start_step_threshold
    args.num_steps_uc = args.num_steps_threshold

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')
    seed = 491

    pl.seed_everything(seed)


    if  args.dataset_name == 'imagenet64':
        model, scheduler = init_guided_diffusion_imagenet64_and_scheduler()
    elif args.dataset_name == 'imagenet128':
        model, scheduler = init_guided_diffusion_imagenet128_and_scheduler()
    else:
        raise ValueError(f'invalid dataset name {args.dataset_name}')
    
    model = model.to(device)
    image_size = int(args.dataset_name.replace('imagenet', ''))
    fid_model = load_real_fid_model(args.dataset_name, device)


    generator = torch.Generator(device=device).manual_seed(seed)
    y = torch.randint(0, 1000, (args.num_samples,), generator=generator, device=device)

    # with open(Path(THRESHOLD / args.dataset_name / f'config_{args.scheduler_type}.yaml'), 'r') as f:
    #     config_threshold = yaml.safe_load(f)
    # dataset_folder = config_threshold['dataset_folders'][0]
    # dataset_folder = Path(dataset_folder)
    # with open(dataset_folder / 'args.yaml', 'r') as f:
    #     config_generation = yaml.safe_load(f)
    # generation_steps = config_generation['generation_steps']
    # print(config_generation.keys())

    # for k, v in config_generation.items():
    #     if k not in args.__dict__:
    #         args.__dict__[k] = v
    #         setattr(args, k, v)
    # if 'start_step_uc' not in args.__dict__:
    #     args.start_step_uc = 0
    # if 'num_steps_uc' not in args.__dict__:
    #     args.num_steps_uc = generation_steps

    ddim_sampler = DDIMScheduler.from_config(scheduler.config)
    ddim_sampler.set_timesteps(args.num_steps)

    uc_sampler = instantiate_uc_scheduler(args, scheduler, model, y)
    uc_sampler.set_timesteps(args.num_steps)

    fid = load_real_fid_model(args.dataset_name, device)

    # gen_images_threshold = generate_samples_model_scheduler_class_conditioned_with_threshold(
    #     num_samples=args.num_samples,
    #     batch_size=args.batch_size,
    #     image_size=image_size,
    #     model=model,
    #     scheduler=uc_sampler,
    #     num_classes=y,
    #     threshold=thresholds,
    #     fid_evaluator=fid,
    #     device=device,
    #     start_step=args.start_step_threshold,
    #     num_steps=args.num_steps_threshold,
    #     seed=seed,
    # )

    gen_images_threshold = generate_samples_model_scheduler_class_conditioned_with_percentile(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=image_size,
        model=model,
        scheduler=uc_sampler,
        num_classes=y,
        percentile=args.percentile,
        fid_evaluator=fid,
        device=device,
        start_step=args.start_step_threshold,
        num_steps=args.num_steps_threshold,
        seed=seed,
    )


    print('FID of samples with threshold', gen_images_threshold['fid'])

    fid = load_real_fid_model(args.dataset_name, device)

    gen_images = generate_samples_model_scheduler_class_conditioned(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=image_size,
        model=model,
        scheduler=ddim_sampler,
        num_classes=y,
        fid_evaluator=fid,
        device=device   
    )


    print('FID of samples without threshold', gen_images['fid'])

    print('Delta FID', gen_images['fid'] - gen_images_threshold['fid'])

def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num-samples', type=int, default=1000, help='number of samples to generate')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    argparser.add_argument('--scheduler-type', '--scheduler', dest='scheduler_type', type=str, default='uncertainty')
    argparser.add_argument('--dataset-folder', '--dataset', type=str, required=True, help='path to the dataset folder containing the uncertainty and gen_images files', dest='dataset_name')
    argparser.add_argument('--start-step-threshold', type=int, default=0, help='step to start estimating the threshold')
    argparser.add_argument('--num-steps-threshold', type=int, default=20, help='number of steps to estimate the threshold')
    argparser.add_argument('--percentile', type=float, default=0.95, help='percentile for the threshold')
    argparser.add_argument('--num-steps', type=int, default=20, help='number of steps to generate')

    options = CONFIG.joinpath('generation').files('*.yaml')
    options = [x.basename().replace('.yaml', '') for x in options]

    argparser.add_argument('--config', type=str, default=None, help='path to the config file', choices=options)
    
    args = argparser.parse_args()

    if args.config is not None:
        print('Loading config file - ignoring other arguments')
        args = load_config(CONFIG / 'generation', args.config)

    return args



if __name__ == '__main__':
    main()