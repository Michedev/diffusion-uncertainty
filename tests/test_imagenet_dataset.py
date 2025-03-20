import pickle
from diffusion_uncertainty.dataset.imagenet import ImagenetDataset
from diffusion_uncertainty.paths import IMAGENET128
import numpy as np

def test_labels_imagenet():
    dataset = ImagenetDataset(IMAGENET128, 'train')

    class_idxs = dataset.classes.values()

    assert len(class_idxs) == 1000, len(class_idxs)

    assert min(class_idxs) == 0, min(class_idxs)
    assert max(class_idxs) == 999, max(class_idxs)


def test_labels_imagenet_val():
    dataset = ImagenetDataset(IMAGENET128, 'val')

    class_idxs = dataset.classes

    assert len(class_idxs) == 50000, len(class_idxs)

    assert min(class_idxs) == 0, min(class_idxs)
    assert max(class_idxs) == 999, max(class_idxs)

def test_load_imagenet64():
    path_val_npz = 'data/imagenet64/Imagenet64_val_npz/val_data.npz'

    with np.load(path_val_npz) as f:
        keys_data = f.keys()
        keys_data = list(keys_data)
        data = f['data']

    assert keys_data == ['data', 'labels'], keys_data 

    assert data.shape == (50000, 3 * 64 * 64), data.shape

    assert data.dtype == np.uint8, data.dtype
    assert data.min() >= 0, data.min()
    assert data.max() <= 255, data.max()
