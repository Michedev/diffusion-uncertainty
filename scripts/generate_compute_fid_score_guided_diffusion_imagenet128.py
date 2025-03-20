import argparse
import sys
from path import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from diffusion_uncertainty.paths import INTERMEDIATES
from diffusion_uncertainty.generate_samples import generate_samples_and_compute_fid_model_scheduler_class_conditioned


from diffusion_uncertainty.init_model import init_guided_diffusion_imagenet128_and_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertainty as DDIMSchedulerUncertaintyV4, DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedV4
from diffusion_uncertainty.generate_samples import generate_samples_and_compute_fid_model_scheduler
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlip
from diffusion_uncertainty.fid import load_real_fid_model
import torch
import pytorch_lightning as pl

@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num-samples', type=int, default=300, dest='num_samples')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--generation-steps', type=int, default=20, dest='generation_steps')
    argparser.add_argument('--M', type=int, default=30, dest='M')
    argparser.add_argument('--after-step', type=int, default=0)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--eta', type=float, default=0.00)
    argparser.add_argument('--uncertainty-steps', type=int, default=20)
    argparser.add_argument('--condition-score', action='store_true', dest='condition_score')
    argparser.add_argument('--predict-next', action='store_true', dest='predict_next')

    argparser.add_argument('--scheduler', type=str, default='uncertainty', choices=['uncertainty', 'flip'], dest='scheduler_type')

    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')

    argparser.add_argument('--store-intermediate', action='store_true', dest='store_intermediate')
    argparser.add_argument('--load-intermediate', action='store_true', dest='load_intermediate')

    args = argparser.parse_args()

    if args.num_samples % args.batch_size != 0:
        args.num_samples = args.num_samples + (args.batch_size - args.num_samples % args.batch_size)
        print(f"Adjusted number of samples to {args.num_samples} to be divisible by batch size {args.batch_size}")
    
    pl.seed_everything(args.seed)

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')

    unet, scheduler = init_guided_diffusion_imagenet128_and_scheduler()

    unet.to(device)

    y = torch.randint(0, 1000, (args.num_samples,)).to(device)

    if args.scheduler_type == 'uncertainty':
        uc_scheduler = DDIMSchedulerUncertaintyImagenetClassConditionedV4.from_config(scheduler.config, M=args.M, after_step=args.after_step, num_steps_uc=args.uncertainty_steps, update_score_uc=args.condition_score, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'flip':
        uc_scheduler = DDIMSchedulerUncertaintyFlip.from_config(scheduler.config, M=args.M, after_step=args.after_step, num_steps_uc=args.uncertainty_steps, unet=unet, eta=args.eta, prompt_embeds=y)
    uc_scheduler.set_timesteps(args.generation_steps)

    fid_evaluator = load_real_fid_model('imagenet128', device, normalize=False)

    ddim_scheduler = DDIMScheduler.from_config(scheduler.config, eta=args.eta)
    ddim_scheduler.set_timesteps(args.generation_steps)

    pl.seed_everything(args.seed)
    output = generate_samples_and_compute_fid_model_scheduler_class_conditioned(args.num_samples, args.batch_size, fid_evaluator, image_size=128, model=unet, scheduler=uc_scheduler, device=device, num_classes=y, return_intermediates=args.store_intermediate, load_intermediate=args.load_intermediate)

    suffix = '_conditioned' if args.condition_score else ''
    suffix += '_predict_next' if args.predict_next else '_predict_x0'

    #todo: implement load_intermediate
    if not INTERMEDIATES.joinpath('imagenet128').exists():
        INTERMEDIATES.joinpath('imagenet128').mkdir()
    if args.store_intermediate:
        fid_score_uc, intermediates = output
        torch.save(intermediates['uncertainty'], INTERMEDIATES / 'imagenet128' / f"uncertainty_seed={args.seed}_M={args.M}_steps={args.generation_steps}_scheduler={args.scheduler_type}{suffix}.pth")
        torch.save(intermediates['y'], INTERMEDIATES / 'imagenet128' / f"y_seed={args.seed}_M={args.M}_steps={args.generation_steps}_scheduler={args.scheduler_type}{suffix}.pth")
        torch.save(intermediates['x_t'], INTERMEDIATES / 'imagenet128' / f"x_t_seed={args.seed}_M={args.M}_steps={args.generation_steps}_scheduler={args.scheduler_type}{suffix}.pth")
        torch.save(intermediates['score'], INTERMEDIATES / 'imagenet128' / f"score_seed={args.seed}_M={args.M}_steps={args.generation_steps}_scheduler={args.scheduler_type}{suffix}.pth")

        print('Stored intermediate results in', INTERMEDIATES / 'imagenet128')
    else:
        fid_score_uc = output
    if not args.condition_score:
        pl.seed_everything(args.seed)

        fid_score = generate_samples_and_compute_fid_model_scheduler_class_conditioned(args.num_samples, args.batch_size, fid_evaluator, image_size=128, model=unet, scheduler=ddim_scheduler, device=device, num_classes=y)

        print(f"FID Score normal model: {fid_score}")
        print(f"FID Score uncertainty model: {fid_score_uc}")
        print(f"Improvement: {fid_score - fid_score_uc}")


if __name__ == '__main__':
    main()