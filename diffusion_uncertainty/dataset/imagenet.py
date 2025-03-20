
from email.mime import image
from typing import Literal
from path import Path
from torch.utils.data import Dataset
from beartype import beartype
from PIL import Image
from torchvision import transforms
from diffusion_uncertainty.paths import IMAGENET_CLASS_MAP, IMAGENET_VALIDATION_GROUND_TRUTH

class ImagenetDataset(Dataset):

    @beartype
    def __init__(self, root: Path, split: Literal['train', 'val', 'test'], transform=None):
            """
            Initialize the Imagenet dataset.

            Args:
                root (Path): The root directory of the dataset.
                split (Literal['train', 'val', 'test']): The split of the dataset ('train', 'val', or 'test').
                transform: Optional transform to be applied on a sample.

            """
            self.root = root
            self.transform = transform
            self.split = split

            self.root_split = self.root / self.split

            if self.split == 'train':
                # Get the directories for training split
                root_split1 = self.root_split / '3' / 'box'
                root_split2 = self.root_split / 'box'
                self.dirs = []
                if root_split1.exists():
                    self.dirs += root_split1.dirs()
                if root_split2.exists():
                    self.dirs += root_split2.dirs()
                # self.dirs = root_split1.dirs() + root_split2.dirs()
                self.dir_names: set[Path] = {d.name for d in self.dirs}

                # Read the class map file
                with open(IMAGENET_CLASS_MAP, 'r') as f:
                    lines = f.readlines()
                self.classes = {l.split(' ')[0]: int(l.split(' ')[1].strip())-1 for l in lines}

                print('num dir_names', len(self.dir_names))

                # Get the list of image files
                self.files = []
                for d in self.dirs:
                    self.files += d.files('*.png')

            elif self.split == 'val':
                # Read the validation ground truth file
                with open(IMAGENET_VALIDATION_GROUND_TRUTH, 'r') as f:
                    lines = f.readlines()

                # Get the list of image files for validation split
                self.files = self.root_split.joinpath('box').files('*.png')
                self.files = sorted(self.files, key=lambda x: int(x.basename().replace('ILSVRC2012_val_000', '').replace('.png', '')))

                self.classes = [int(l.strip())-1 for l in lines]

            else:
                # Get the list of image files for test split
                self.files = self.root_split.joinpath('box').files('*.png')
            if self.transform is None:
                self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    
    # def __getitem__(self, idx) for train
    def getitem_with_class(self, idx):
        """
        Retrieves the image and corresponding class label at the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding class label.
        """
        img = Image.open(self.files[idx])
        y = self.classes[self.files[idx].parent.name]
        if self.transform:
            img = self.transform(img)
        return dict(image=img, label=y)
    
    def getitem_with_class_idx(self, idx):
        """
        Retrieves the image and class index at the given index.

        Args:
            idx (int): The index of the image and class index to retrieve.

        Returns:
            tuple: A tuple containing the image and class index.
        """
        img = Image.open(self.files[idx])
        y = self.classes[idx]
        if self.transform:
            img = self.transform(img)
        return dict(image=img, label=y)

    # def __getitem__(self, idx) for val and test
    def getitem_without_class(self, idx):
        img = Image.open(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return dict(image=img)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            return self.getitem_with_class(idx)
        elif self.split == 'val':
            return self.getitem_with_class_idx(idx)
        else:
            return self.getitem_without_class(idx)