from dataclasses import dataclass
from typing import Literal, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
from beartype import beartype
from diffusion_uncertainty.fid import load_real_fid_model
from diffusion_uncertainty.paths import DATASET_FID
from diffusers.pipelines.ddim.pipeline_ddim import DDIMPipeline
from diffusers.schedulers import SchedulerMixin, DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from scripts.eval_fid_lsun_churches256 import generate_samples_and_compute_fid
from diffusion_uncertainty.generate_samples import generate_samples_and_compute_fid_model_scheduler, generate_samples_and_compute_fid_model_scheduler_class_conditioned

class DDPMFinetuneNoise(pl.LightningModule):

    @beartype
    def __init__(self, unet, noise_schedule: DDPMScheduler, M: int, lr=1e-4, warmup_step: int = 500, time_zone: Literal['start', 'middle', 'end', 'all'] = 'all', class_conditioned: bool = False, use_lr_scheduler: bool = False, mode: Literal['min', 'max'] = 'min', optimizer: Literal['adam', 'adamw', 'sgd', 'rmsprop'] = 'adam', momentum: float = 0.9, beta_1: float = 0.9, beta_2: float = 0.999, weight_decay: float = 0.0, model_id: Optional[str] = None, dataset_name: Optional[str] = None):
        """
        Initializes the DDPMFinetuneNoise class.

        Args:
            unet: The U-Net model.
            noise_schedule: The noise schedule.
            M (int): The number of noise samples.
            lr (float, optional): The learning rate. Defaults to 1e-4.
            warmup_step (int, optional): The number of warm-up steps. Defaults to 500.
            time_zone (Literal['start', 'middle', 'end', 'all'], optional): The time zone for training the diffusion model. Defaults to 'all'.
            use_lr_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
            mode (Literal['min', 'max'], optional): Whether to minimize or maximize the uncertainty. Defaults to 'min'.
            optimizer (str, optional): The optimizer to use. Defaults to 'adam'.
        """
        super().__init__()

        self.unet = unet
        self.noise_schedule = noise_schedule
        self.lr = lr
        self.M = M
        self.warmup_step = warmup_step
        self.time_zone = time_zone
        if time_zone == 'start':
            self.start_step = 0
            self.end_step = self.noise_schedule.num_train_timesteps // 4
        elif time_zone == 'middle':
            self.start_step = self.noise_schedule.num_train_timesteps // 4
            self.end_step = self.noise_schedule.num_train_timesteps * 3 // 4
        elif time_zone == 'end':
            self.start_step = self.noise_schedule.num_train_timesteps * 3 // 4
            self.end_step = self.noise_schedule.num_train_timesteps
        elif time_zone == 'all':
            self.start_step = 0
            self.end_step = self.noise_schedule.num_train_timesteps
        print(f"Training on timesteps {self.start_step} to {self.end_step} - Total timesteps: {self.noise_schedule.num_train_timesteps}")
        self.use_lr_scheduler = use_lr_scheduler
        self.mode = mode
        self.optimizer_name = optimizer
        self.momentum = momentum
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.class_conditioned = class_conditioned

        self.save_hyperparameters('lr', 'M', 'warmup_step', 'time_zone', 'use_lr_scheduler', 'mode', 'optimizer', 'momentum', 'weight_decay', 'model_id', 'dataset_name', 'beta_1', 'beta_2', 'class_conditioned', 'noise_schedule')


    def forward(self, x, t, y=None):
        return self.unet(x, t)
    
    def training_step(self, batch, batch_idx):
        if self.class_conditioned:
            X, y = batch
        else:
            X= batch if not isinstance(batch, tuple) else batch[0]
            y = None
        bs = X.shape[0]

        timesteps = torch.randint(self.start_step, self.end_step, (bs,), device=X.device, dtype=torch.long)

        with torch.no_grad():
            best_noise = torch.zeros_like(X)
            best_uncertainty = torch.zeros(bs, 1, 1, 1, device=X.device)
            if self.mode == 'min':
                best_uncertainty.fill_(float('inf'))
            for _ in range(self.M):
                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noise = torch.randn_like(X)
                noisy_images = self.noise_schedule.add_noise(X, noise, timesteps)

                # Compute the log likelihood of the noisy images under the model
                eps = self(noisy_images, timesteps, y=y)

                noisy_images_flip = torch.flip(noisy_images, dims=(2,))

                eps_flip = self(noisy_images_flip, timesteps, y=y)

                uncertainty = (eps - eps_flip).pow(2).sum(dim=(1, 2, 3), keepdim=True)
                if self.mode == 'min':
                    is_better = uncertainty < best_uncertainty
                else:
                    is_better = uncertainty > best_uncertainty
                best_noise = torch.where(is_better, noise, best_noise)
                best_uncertainty = torch.where(is_better, uncertainty, best_uncertainty)

        del noise, noisy_images, eps, eps_flip, uncertainty, is_better, best_uncertainty

        # Add noise to the clean images according to the noise magnitude at each timestep

        noisy_images = self.noise_schedule.add_noise(X, best_noise, timesteps)

        eps = self(noisy_images, timesteps, y=y)

        loss = (eps - best_noise).pow(2).mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.class_conditioned:
            X, y = batch
        else:
            X= batch if not isinstance(batch, tuple) else batch[0]
            y = None
        bs = X.shape[0]

        noise = torch.randn_like(X)

        timesteps = torch.randint(0, self.noise_schedule.num_train_timesteps, (bs,), device=X.device, dtype=torch.long)

        noisy_images = self.noise_schedule.add_noise(X, noise, timesteps)

        eps = self(noisy_images, timesteps, y=y)

        loss = (eps - noise).pow(2).mean(dim=0).sum()

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        FID_DATASET = DATASET_FID / self.dataset_name

        lsun_fid = load_real_fid_model(FID_DATASET, device=self.device, normalize=True)
        
        # scheduler = DDIMScheduler.from_config(self.noise_schedule.config)
        # # scheduler.to(self.device)
        
        pipeline = DDIMPipeline.from_pretrained(self.model_id)
        pipeline.to(self.device)

        batch_size = 32
        num_samples = 300
        generation_steps = 20
        

        fid_score = generate_samples_and_compute_fid(num_samples, generation_steps, batch_size, lsun_fid, pipeline, device=self.device)

        self.log('fid_score', fid_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta_1, self.beta_2))
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=self.momentum)
        
        optimizers_dict = {"optimizer": optimizer}
        if self.use_lr_scheduler:
            # CosineAnnealingWarmRestarts scheduler with warmup
            warmup_step = self.warmup_step
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_step, T_mult=2, eta_min=0)
            optimizers_dict["lr_scheduler"] = scheduler
        
        return optimizers_dict


@dataclass
class DummyConfig:

    in_channels: int
    sample_size: int

class DDPMFinetuneNoiseImagenet(DDPMFinetuneNoise):

    first_fid = True
    original_model_fid = None

    def forward(self, x, t, y=None):
        if self.class_conditioned:
            eps = self.unet(x, t, y=y)
        else:
            eps = self.unet(x, t) 
        eps = eps[:, :3]
        return eps
    

    def on_validation_epoch_end(self) -> None:
        image_size = int(self.dataset_name.replace('imagenet', ''))
 
        FID_DATASET = DATASET_FID / self.dataset_name

        self.unet = self.unet.float()

        imagenet_fid = load_real_fid_model(FID_DATASET, device=self.device, normalize=False)

        imagenet_fid = imagenet_fid.to(self.device)
        
        scheduler = DDIMScheduler.from_config(self.noise_schedule.config)
        # scheduler.to(self.device)
        
        batch_size = 32
        num_samples = 300
        generation_steps = 20

        scheduler.set_timesteps(generation_steps)

        fid_score = generate_samples_and_compute_fid_model_scheduler_class_conditioned(num_samples, generation_steps, batch_size, imagenet_fid, image_size=image_size, model=self.unet,  num_classes=1000, scheduler=scheduler, device=self.device)

        if self.first_fid:
            print('FID model not finetuned:', fid_score)
            self.first_fid = False
            self.original_model_fid = fid_score
        else:
            improvement_fid = self.original_model_fid - fid_score
            print('FID model finetuned:', fid_score)
            print('Improvement:', improvement_fid)
            self.log('fid_score_improvement', improvement_fid, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log('fid_score', fid_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return fid_score

