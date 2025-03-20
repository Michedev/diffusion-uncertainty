from typing import Literal
from beartype import beartype
from torchvision import transforms
from path import Path
from torch.utils.data import Dataset
from PIL import Image


class CIFAR10Dataset(Dataset):
    cifar10_class_names = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    cifar10_class_indexes = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    @beartype
    def __init__(self, root_path: str | Path, split: Literal['train', 'test']):
        """
        Initialize the CIFAR-10 dataset.

        Args:
            path (str): The path to the dataset.
            split (Literal['train', 'test']): The split of the dataset ('train' or 'test').
        """
        assert root_path.exists(), f'{root_path} does not exist'
        assert root_path.joinpath('images').exists(), f'{root_path}/images does not exist'
        assert root_path.joinpath('images').joinpath(split).exists(), f'{root_path}/images/{split} does not exist'
        for i in range(10):
            basepath = root_path.joinpath('images').joinpath(split)
            assert basepath.joinpath(self.cifar10_class_names[i].capitalize()).exists(), f'{root_path}/images/{split}/{self.cifar10_class_names[i].capitalize()} does not exist'
        self.path = root_path
        self.split = split
        self.image_paths = list(self.path.joinpath('images').joinpath(split).walkfiles('*.png')) + list(self.path.joinpath('images').joinpath(split).walkfiles('*.jpg'))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx])
        image = self.to_tensor(image)
        class_name = self.image_paths[idx].parent.basename().lower()
        y = self.cifar10_class_indexes[class_name]
        return {'image': image, 'label': y}