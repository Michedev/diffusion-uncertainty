from typing import Literal
import torch
from diffusion_uncertainty.paths import LSUN_CHURCHES256_TRAIN, LSUN_CHURCHES256_VAL
from torchvision import transforms
from PIL import Image

class LSUNChurches256(torch.utils.data.Dataset):
    def __init__(self, split: Literal['train', 'val'] = 'train', transforms=None):
        super().__init__()
        assert split in ['train', 'val']
        self.split = split
        if split == 'train':
            self.root = LSUN_CHURCHES256_TRAIN
        else:
            self.root = LSUN_CHURCHES256_VAL
        self.files = list(self.root.files('*.webp'))
        if transforms is None:
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        img = self.transforms(img)


        return img