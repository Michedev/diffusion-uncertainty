import torch
from diffusion_uncertainty.guided_diffusion.unet_openai import UNetModel
from diffusion_uncertainty.paths import MODELS


def test_load_imagenet64():

    hparams = dict(
        image_size=64, in_channels=3, model_channels=192, out_channels=6, num_res_blocks=3, attention_resolutions=(2, 4, 8), dropout=0.1, channel_mult=(1, 2, 3, 4), num_classes=1000, use_checkpoint=False, use_fp16=True, num_heads=4, num_head_channels=64, num_heads_upsample=4, use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=True
    )

    model = UNetModel(**hparams)

    model.load_state_dict(torch.load(MODELS / f"64x64_diffusion.pt"))