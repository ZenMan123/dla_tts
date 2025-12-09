import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class HiFiGANGeneratorLoss(nn.Module):
    def __init__(self, audio_to_mel, lambda_fm=2.0, lambda_mel=45.0, device="cuda"):
        super().__init__()
        self.audio_to_mel = audio_to_mel.to(torch.device(device))
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

    def forward(self, batch: dict) -> dict:
        fake = batch["audio_pred"]
        mel_target = batch["mel"]

        fake_scores_mpd = batch["fake_scores_mpd"]
        fake_feats_mpd = batch["fake_feats_mpd"]
        fake_scores_msd = batch["fake_scores_msd"]
        fake_feats_msd = batch["fake_feats_msd"]

        loss_adv = 0.0
        for df in fake_scores_mpd:
            loss_adv += torch.mean((df - 1.0) ** 2)
        for df in fake_scores_msd:
            loss_adv += torch.mean((df - 1.0) ** 2)

        real_feats_mpd = batch["real_feats_mpd"]
        real_feats_msd = batch["real_feats_msd"]

        loss_fm = 0.0

        for rf_list, ff_list in zip(real_feats_mpd, fake_feats_mpd):
            for rf, ff in zip(rf_list, ff_list):
                min_h = min(rf.size(2), ff.size(2))
                loss_fm += torch.mean(
                    torch.abs(rf[:, :, :min_h, :] - ff[:, :, :min_h, :])
                )

        for rf_list, ff_list in zip(real_feats_msd, fake_feats_msd):
            for rf, ff in zip(rf_list, ff_list):
                min_len = min(rf.size(-1), ff.size(-1))
                loss_fm += torch.mean(torch.abs(rf[..., :min_len] - ff[..., :min_len]))

        real_mel = mel_target
        fake_mel = self.audio_to_mel(fake.squeeze(1))

        min_T = min(real_mel.size(-1), fake_mel.size(-1))
        real_mel = real_mel[..., :min_T]
        fake_mel = fake_mel[..., :min_T]

        loss_mel = F.l1_loss(fake_mel, real_mel)

        loss = loss_adv + self.lambda_fm * loss_fm + self.lambda_mel * loss_mel

        return {
            "loss_generator": loss,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm,
            "loss_mel": loss_mel,
        }
