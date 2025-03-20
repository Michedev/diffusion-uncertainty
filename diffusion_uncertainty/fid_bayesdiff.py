import torch

class FIDBayesDiff:

    def __init__(self, device: torch.device, dataset_name: str) -> None:
        self.device = device
        self.real_embeddings = []
        self.fake_embeddings = []
        self.dataset_name = dataset_name
        

    def update(self, images: torch.Tensor, real: bool) -> None:
        """
        Update the FID evaluator with new images.

        Args:
            images (torch.Tensor): The images to update the evaluator with.
            real (bool): Whether the images are real or fake.
        """
        if real:
            self.real_embeddings.append(images)
        else:
            self.fake_embeddings.append(images)

    