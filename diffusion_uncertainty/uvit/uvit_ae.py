import torch


class UViTAE(torch.nn.Module):
    def __init__(self, uvit, autoencoder):
        super().__init__()
        self.uvit = uvit
        self.autoencoder = autoencoder
        self.in_chans = uvit.in_chans
        self.img_size = uvit.img_size
        self.patch_size = uvit.patch_size
        self.embed_dim = uvit.embed_dim


    def forward(self, x, t, y):
        x = self.uvit(x, t, y)
        return x
    
    def decode(self, x):
        return self.autoencoder.decode(x)
    
    def to(self, device):
        self.uvit.to(device)
        self.autoencoder.to(device)
        return self