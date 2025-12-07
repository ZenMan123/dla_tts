import torch
import torchaudio


class DeterministicModel:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=1.0,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.mel_transform.spectrogram.power = power

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(audio).to(self.device).clamp_(min=1e-5).log_()
        return mel
