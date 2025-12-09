import torch.nn as nn

from src.loss.discriminator_loss import HiFiGANDiscriminatorLoss
from src.loss.generator_loss import HiFiGANGeneratorLoss


class HiFiGANLoss(nn.Module):
    def __init__(self, audio_to_mel):
        super().__init__()
        self.generator_loss = HiFiGANGeneratorLoss(audio_to_mel)
        self.discriminator_loss = HiFiGANDiscriminatorLoss()

    def forward(self, **batch):
        gen_losses = self.generator_loss(batch)
        disc_losses = self.discriminator_loss(batch)

        return {
            **gen_losses,
            **disc_losses,
        }
