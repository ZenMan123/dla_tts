import torch
import torch.nn as nn
from nemo.collections.tts.models import FastPitchModel


class AcousticModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = (
            FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        tokens = self.model.parse(text).to(self.device)
        mel = self.model.generate_spectrogram(tokens=tokens)
        mel = mel + 4.4
        return mel
