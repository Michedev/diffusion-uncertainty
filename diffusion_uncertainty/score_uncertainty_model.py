from typing import Literal
import pytorch_lightning as pl
from torch import nn
import torch

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ScoreUncertaintyModel(pl.LightningModule):

    def __init__(self, in_channels: int, num_timesteps: int, image_size: int, optimizer: Literal['adam', 'sgd'], learning_rate: float):
        super().__init__()

        # Note: I tried to use a ResNet architecture here, but I get the same results as this simple architecture, so I'm sticking with this one for now.
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        self.embedding_channels = 1

        self.timestep_embedding = nn.Embedding(num_timesteps, self.embedding_channels * image_size * image_size) #todo check embedding layer definition

        self.output_norm = nn.GroupNorm(num_groups=1, num_channels=self.embedding_channels, eps=1e-8)

        self.mse = nn.MSELoss(reduction='none')

        self.in_channels = in_channels
        self.image_size = image_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
            
    def forward(self, score, timestep):
        output = self.conv_net(score)  + self.timestep_embedding(timestep).view(-1, self.embedding_channels, self.image_size, self.image_size)

        output = self.output_norm(output)

        return output
    
    def training_step(self, batch, batch_idx):
        score, timestep, uncertainty = batch

        pred_uc = self(score, timestep)

        loss = self.mse(uncertainty, pred_uc)

        loss = loss.mean(dim=0).sum()

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        score, timestep, uncertainty = batch

        pred_uc = self(score, timestep)

        loss = self.mse(uncertainty, pred_uc)

        loss = loss.mean(dim=0).sum()

        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss
