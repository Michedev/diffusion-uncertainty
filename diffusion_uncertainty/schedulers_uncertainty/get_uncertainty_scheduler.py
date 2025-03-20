from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlip
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_flip_grad import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyFlipGrad
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertainty
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_centered_d import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyCenteredD
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_image import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyImage
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyOriginal
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_uncertainty_zigzag_centered import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyUncertaintyZigZagCentered
from diffusion_uncertainty.schedulers_uncertainty.scheduling_ddim_mc_dropout import DDIMSchedulerUncertaintyImagenetClassConditioned as DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout
from diffusion_uncertainty.schedulers_uncertainty.scheduling_dpm_2_uncertainty_centered import KDPM2SchedulerUncertaintyImagenetClassConditioned as DPM2UncertaintyCentered


def get_uncertainty_scheduler(args, y, unet, scheduler):
    if args.scheduler_type == 'uncertainty':
        uc_scheduler = DDIMSchedulerUncertaintyUncertainty.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_image':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyImage.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'flip':
        uc_scheduler = DDIMSchedulerUncertaintyFlip.from_config(scheduler.config, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, eta=args.eta, prompt_embeds=y)
    elif args.scheduler_type == 'flip_grad':
        uc_scheduler = DDIMSchedulerUncertaintyFlipGrad.from_config(scheduler.config, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, eta=args.eta, prompt_embeds=y)
    elif args.scheduler_type == 'uncertainty_centered':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyCentered.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=args.predict_next, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_original':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyOriginal.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, predict_next=False, y=y, eta=args.eta)
    elif args.scheduler_type == 'uncertainty_centered_d':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyCenteredD.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, y=y, eta=args.eta, uncertainty_distance=args.uncertainty_distance)
    elif args.scheduler_type == 'uncertainty_zigzag_centered':
        uc_scheduler = DDIMSchedulerUncertaintyUncertaintyZigZagCentered.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, y=y, eta=args.eta,
        num_zigzag=args.num_zigzag)
    elif args.scheduler_type == 'dpm_2_uncertainty_centered':
        uc_scheduler = DPM2UncertaintyCentered.from_config(scheduler.config, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, unet=unet, y=y, eta=args.eta)
    else:
        uc_scheduler = DDIMSchedulerUncertaintyImagenetClassConditionedMCDropout.from_config(scheduler.config, prompt_embeds=y, unet=unet, M=args.M, after_step=args.start_step_uc, num_steps_uc=args.num_steps_uc, eta=args.eta)
    return uc_scheduler

# aliases

instatiate_uc_scheduler = get_uncertainty_scheduler
instatiate_uncertainty_scheduler = get_uncertainty_scheduler
