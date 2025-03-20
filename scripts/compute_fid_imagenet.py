import argparse
import json
from random import randint
import sys
from path import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from diffusion_uncertainty.paths import INTERMEDIATES, RESULTS
from diffusion_uncertainty.generate_samples import generate_samples_and_compute_fid_model_scheduler_class_conditioned
from diffusion_uncertainty.score_uncertainty_model import ScoreUncertaintyModel


from diffusion_uncertainty.init_model import init_guided_diffusion_imagenet128_and_scheduler, init_guided_diffusion_imagenet64_and_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertainty as DDIMSchedulerUncertaintyV4, DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedV4
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout
from diffusion_uncertainty.generate_samples import generate_samples_model_scheduler_class_conditioned
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlip
from diffusion_uncertainty.fid import load_real_fid_model
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip_grad import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipGrad
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_grad import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyGrad
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_threshold import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyThreshold
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_score_uncertainty_model_gradient import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyScoreModelGradient
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout_gradient import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyMCDropoutGradient
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_multiscale_threshold import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyThresholdMultiscale
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip_threshold import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipThreshold
import torch
import pytorch_lightning as pl

def instantiate_uc_scheduler(scheduler_type, unet, scheduler, M, after_step, uncertainty_steps, eta, prompt_embeds, device, predict_next, condition_score, uncertainty_threshold=None, uncertainty_threshold_mode=None, uncertainty_normalize=None, score_model_path=None, image_size=None, num_timesteps=None):
    if scheduler_type == 'uncertainty':
        uc_scheduler = DDIMSchedulerUncertaintyImagenetClassConditionedV4.from_config(scheduler.config, M=M, after_step=after_step, num_steps_uc=uncertainty_steps, update_score_uc=False, unet=unet, predict_next=predict_next, y=prompt_embeds, eta=eta)
    elif scheduler_type == 'flip':
        uc_scheduler = DDIMSchedulerUncertaintyFlip.from_config(scheduler.config, M=M, after_step=after_step, num_steps_uc=uncertainty_steps, unet=unet, eta=eta, prompt_embeds=prompt_embeds)
    elif scheduler_type == 'flip_threshold':
        uc_scheduler = DDIMSchedulerUncertaintyFlipThreshold.from_config(scheduler.config, after_step=after_step, num_steps_uc=uncertainty_steps, unet=unet, eta=eta, prompt_embeds=prompt_embeds, uncertainty_threshold=uncertainty_threshold, uncertainty_threshold_mode=uncertainty_threshold_mode, uncertainty_normalize=uncertainty_normalize)
    elif scheduler_type == 'mcdropout':
        assert image_size == 64, "mcdropout only implemented for 64x64"
        uc_scheduler = DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout.from_config(scheduler.config, prompt_embeds=prompt_embeds, unet=unet, M=M, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta)
    elif scheduler_type == 'flip_grad':
        uc_scheduler = DDIMSchedulerUncertaintyFlipGrad.from_config(scheduler.config, after_step=after_step, num_steps_uc=uncertainty_steps, unet=unet, eta=eta, prompt_embeds=prompt_embeds)
    elif scheduler_type == 'uncertainty_grad':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyGrad.from_config(scheduler.config, M=M, unet=unet, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta, prompt_embeds=prompt_embeds, predict_next=predict_next)
    elif scheduler_type == 'uncertainty_threshold':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyThreshold.from_config(scheduler.config, M=M, unet=unet, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta, predict_next=predict_next, prompt_embeds=prompt_embeds, uncertainty_threshold=uncertainty_threshold, uncertainty_threshold_mode=uncertainty_threshold_mode, uncertainty_normalize=uncertainty_normalize) 
    elif scheduler_type == 'uncertainty_threshold_multiscale':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyThresholdMultiscale.from_config(scheduler.config, M=M, unet=unet, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta, predict_next=predict_next, prompt_embeds=prompt_embeds, uncertainty_normalize=uncertainty_normalize)
    elif scheduler_type == 'score_uncertainty_model_gradient':
        assert score_model_path is not None
        score_model = ScoreUncertaintyModel.load_from_checkpoint(score_model_path, image_size=image_size, in_channels=3, num_timesteps=num_timesteps).to(device)
        uc_scheduler = DDIMSchedulerUncertaintyScoreModelGradient.from_config(scheduler.config, score_model=score_model, unet=unet, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta, prompt_embeds=prompt_embeds)
    elif scheduler_type == 'mc_dropout_gradient':
        assert image_size == 64, "mc_dropout_gradient only implemented for 64x64"
        uc_scheduler = DDIMSchedulerUncertaintyMCDropoutGradient.from_config(scheduler.config, prompt_embeds=prompt_embeds, unet=unet, M=M, after_step=after_step, num_steps_uc=uncertainty_steps, eta=eta)
    return uc_scheduler


@torch.no_grad()
def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num-samples', type=int, default=300, dest='num_samples')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--generation-steps', type=int, default=20, dest='generation_steps')
    argparser.add_argument('--M', type=int, default=30, dest='M')
    argparser.add_argument('--after-step', type=int, default=0)
    argparser.add_argument('--seed', type=int, default=None)
    argparser.add_argument('--eta', type=float, default=0.00)
    argparser.add_argument('--uncertainty-steps', type=int, default=20)
    argparser.add_argument('--dropout', type=float, default=0.1)

    argparser.add_argument('--scheduler', type=str, default='mcdropout', choices=['uncertainty', 'flip', 'mcdropout', 'flip_grad', 'uncertainty_grad', 'uncertainty_threshold', 'flip_threshold', 'uncertainty_threshold_multiscale', 'score_uncertainty_model_gradient', 'mc_dropout_gradient'], dest='scheduler_type')

    argparser.add_argument('--predict-next', action='store_true', dest='predict_next')
    argparser.add_argument('--condition-score', action='store_true', dest='condition_score')

    argparser.add_argument('--on-cpu', action='store_true', dest='on_cpu')
    
    argparser.add_argument('--image-size', type=int, default=64, dest='image_size', choices=[64, 128])

    argparser.add_argument('--skip-ddim-fid', action='store_true', dest='skip_ddim_fid')

    uncertainty_params = argparser.add_argument_group('Infer noise params')
    uncertainty_params.add_argument('--uncertainty-threshold', '--ucth', type=float, default=1.0)
    uncertainty_params.add_argument('--uncertainty-threshold-mode', '--ucthm', type=str, default='max', choices=['max', 'min'])
    uncertainty_params.add_argument('--no-uncertainty-normalize', action='store_false', dest='uncertainty_normalize')

    score_uncertainty_model_gradient_params = argparser.add_argument_group('Score uncertainty model gradient params')

    score_uncertainty_model_gradient_params.add_argument('--score-model-path', type=str, default=None, dest='score_model_path')    

    argparser.add_argument('--skip-duplicate-check', action='store_true', dest='skip_duplicate_check')

    args = argparser.parse_args()


    if args.seed is None:
        args.seed = randint(0, 2**32 - 1)
        print('Set seed to', args.seed)

    if args.num_samples % args.batch_size != 0:
        args.num_samples = args.num_samples + (args.batch_size - args.num_samples % args.batch_size)
        print(f"Adjusted number of samples to {args.num_samples} to be divisible by batch size {args.batch_size}")
    
    if not args.skip_duplicate_check and args.scheduler_type == 'uncertainty_threshold' and abs(args.num_samples - 300) < 50:
        with open(RESULTS / 'uncertainty_threshold_data_imagenet128.json', 'r') as file:
            data = json.load(file)
        for item in data:
            if args.uncertainty_threshold == item['threshold'] and args.uncertainty_threshold_mode == item['mode'] and args.after_step == item['after_step'] and args.uncertainty_steps == item['num_steps']:
                print(f"Found stored duplicate")
                return


    pl.seed_everything(args.seed)

    device = torch.device('cpu') if args.on_cpu else torch.device('cuda')

    if args.image_size == 64:
        unet, scheduler = init_guided_diffusion_imagenet64_and_scheduler(dropout=args.dropout)
    elif args.image_size == 128:
        unet, scheduler = init_guided_diffusion_imagenet128_and_scheduler()
    unet = unet.eval()
    unet.to(device)

    y = torch.randint(0, 1000, (args.num_samples,)).to(device)

    uc_scheduler = instantiate_uc_scheduler(scheduler_type=args.scheduler_type, unet=unet, scheduler=scheduler, M=args.M, after_step=args.after_step, uncertainty_steps=args.uncertainty_steps, eta=args.eta, uncertainty_threshold=args.uncertainty_threshold, uncertainty_threshold_mode=args.uncertainty_threshold_mode, uncertainty_normalize=args.uncertainty_normalize, score_model_path=args.score_model_path, image_size=args.image_size, predict_next=args.predict_next, device=device, condition_score=args.condition_score, prompt_embeds=y)
    
    uc_scheduler.set_timesteps(args.generation_steps)

    fid_evaluator = load_real_fid_model(f'imagenet{args.image_size}', device, normalize=False)

    ddim_scheduler = DDIMScheduler.from_config(scheduler.config, eta=args.eta)
    ddim_scheduler.set_timesteps(args.generation_steps)

    pl.seed_everything(args.seed)
    with torch.autocast('cuda'):
        output = generate_samples_model_scheduler_class_conditioned(args.num_samples, args.batch_size,  image_size=args.image_size, model=unet, scheduler=uc_scheduler, device=device, num_classes=y, fid_evaluator=fid_evaluator)

    fid_uc = output['fid']

    fid_evaluator.reset()
    if not args.skip_ddim_fid:
        print('FID UC:', fid_uc)
        del output

        pl.seed_everything(args.seed)
        with torch.autocast('cuda'):
            output = generate_samples_model_scheduler_class_conditioned(args.num_samples, args.batch_size,  image_size=args.image_size, model=unet, scheduler=ddim_scheduler, device=device, num_classes=y, fid_evaluator=fid_evaluator)

        fid_ddim = output['fid']
        print('FID DDIM:', fid_ddim)
    else:
        if args.image_size == 128 and abs(args.num_samples - 300) < 50:
            fid_ddim = 118.95 # hardcoded after many experiments
        else:
            raise NotImplementedError("FID DDIM not computed for this configuration")

    print(f'Imagenet{args.image_size}')
    print('==============================')
    print(f"FID DDIM: {fid_ddim}")
    print(f"FID UC {args.scheduler_type=}: {fid_uc}")
    print('==============================')
    print(f"Delta FID: {fid_uc - fid_ddim}")

    if args.scheduler_type == 'uncertainty_threshold' and not args.skip_duplicate_check:
        with open(RESULTS / 'uncertainty_threshold_data_imagenet128.json', 'r') as file:
            data = json.load(file)
        if isinstance(fid_uc, torch.Tensor):
            fid_uc = fid_uc.item()
        data.append({
            'threshold': args.uncertainty_threshold,
            'mode': args.uncertainty_threshold_mode,
            'after_step': args.after_step,
            'num_steps': args.uncertainty_steps,
            'FID_DDIM': fid_ddim,
            'FID_UC': fid_uc,
            'Delta_FID': fid_uc - fid_ddim,
        })
        with open(RESULTS / 'uncertainty_threshold_data_imagenet128.json', 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()