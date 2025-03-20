import torch
from diffusion_uncertainty.uvit.uvit import UViT
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.unets import UNet2DModel

class PredictorClassConditionedTrait:

    def predict_model(self, x, t):
        if isinstance(t, float):
            t = round(t)
        if isinstance(t, int):
            t = torch.zeros(size=(x.shape[0],), dtype=torch.int64, device=x.device).fill_(t)
        if isinstance(self.unet, (UViT, UViTAE)):
            return self.unet(x, t, self.prompt_embeds)
        elif isinstance(self.unet, UNet2DModel):
            return self.unet(x, t).sample
        else:
            return self.unet(x, t, y=self.prompt_embeds)[:, :3]