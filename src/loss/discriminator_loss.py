import torch
import torch.nn as nn


class HiFiGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch: dict) -> dict:
        fake_scores_mpd = batch["fake_scores_mpd_detached"]
        fake_scores_msd = batch["fake_scores_msd_detached"]
        real_scores_mpd = batch["real_scores_mpd"]
        real_scores_msd = batch["real_scores_msd"]

        loss_mpd = 0.0
        for dr, df in zip(real_scores_mpd, fake_scores_mpd):
            loss_mpd += torch.mean((dr - 1.0) ** 2) + torch.mean(df**2)

        loss_msd = 0.0
        for dr, df in zip(real_scores_msd, fake_scores_msd):
            loss_msd += torch.mean((dr - 1.0) ** 2) + torch.mean(df**2)

        loss = loss_mpd + loss_msd

        return {
            "loss_discriminator": loss,
            "loss_mpd": loss_mpd,
            "loss_msd": loss_msd,
        }
