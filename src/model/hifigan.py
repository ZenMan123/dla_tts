import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import spectral_norm, weight_norm


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, leaky_relu_coef=0.1):
        super().__init__()
        self.pipes = nn.ModuleList()
        for ds in dilations:
            pipe = []
            for d in ds:
                pipe.append(nn.LeakyReLU(leaky_relu_coef))
                pipe.append(
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            stride=1,
                            padding=get_padding(kernel_size, d),
                            dilation=d,
                        )
                    )
                )
            self.pipes.append(nn.Sequential(*pipe))

    def forward(self, x):
        for pipe in self.pipes:
            x = x + pipe(x)
        return x


class MultiReceptiveField(nn.Module):
    def __init__(
        self,
        channels,
        kernel_sizes=(3, 7, 11),
        dilations=(
            ((1, 1), (3, 1), (5, 1)),
            ((1, 1), (3, 1), (5, 1)),
            ((1, 1), (3, 1), (5, 1)),
        ),
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(channels, k, d_list)
                for k, d_list in zip(kernel_sizes, dilations)
            ]
        )

    def forward(self, x):
        out = 0
        for block in self.blocks:
            out = out + block(x)
        out = out / len(self.blocks)
        return out


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=(
            ((1, 1), (3, 1), (5, 1)),
            ((1, 1), (3, 1), (5, 1)),
            ((1, 1), (3, 1), (5, 1)),
        ),
        h_u=512,
        leaky_relu_coef=0.1,
    ):
        super().__init__()

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channels,
                h_u,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        self.num_upsamples = len(upsample_rates)
        cur_channels = h_u

        self.pipes = nn.ModuleList()

        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            conv_transpose = weight_norm(
                nn.ConvTranspose1d(
                    cur_channels,
                    cur_channels // 2,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

            mrf = MultiReceptiveField(
                channels=cur_channels // 2,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilations,
            )

            self.pipes.append(
                nn.Sequential(
                    nn.LeakyReLU(leaky_relu_coef),
                    conv_transpose,
                    mrf,
                )
            )

            cur_channels = cur_channels // 2

        self.conv_post = weight_norm(
            nn.Conv1d(
                cur_channels,
                1,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )
        self.act = nn.LeakyReLU(leaky_relu_coef)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, mel):
        x = self.conv_pre(mel)
        x = self.act(x)

        for pipe in self.pipes:
            x = pipe(x)

        x = self.conv_post(self.act(x))
        x = torch.tanh(x)
        return x


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

        chs = [1, 32, 128, 512, 1024, 1024, 1]
        layers = []
        for i in range(len(chs) - 2):
            layers.append(
                weight_norm(
                    nn.Conv2d(
                        chs[i],
                        chs[i + 1],
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                )
            )
        layers.append(
            weight_norm(
                nn.Conv2d(
                    chs[-2],
                    chs[-1],
                    kernel_size=(3, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
            )
        )
        self.convs = nn.ModuleList(layers)
        self.act = nn.LeakyReLU(0.1)

    def convert_to_2d(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            t = t + pad_len
        x = x.view(b, c, t // self.period, self.period)
        return x

    def forward(self, x):
        x = self.convert_to_2d(x)

        feats = []
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                x = conv(x)
            else:
                x = self.act(conv(x))
            feats.append(x)

        score = x.view(x.shape[0], -1)
        return score, feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor):
        scores = []
        feats_all = []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            feats_all.append(f)
        return scores, feats_all


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm_f):
        super().__init__()

        chs = [1, 128, 128, 256, 512, 1024, 1024, 1024, 1]
        k_sizes = [15, 41, 41, 41, 41, 41, 5, 3]
        strides = [1, 2, 2, 4, 4, 1, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1, 1]

        convs = []
        for i in range(len(k_sizes)):
            conv = norm_f(
                nn.Conv1d(
                    chs[i],
                    chs[i + 1],
                    kernel_size=k_sizes[i],
                    stride=strides[i],
                    padding=(k_sizes[i] - 1) // 2,
                    groups=groups[i],
                )
            )
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        feats = []
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                x = conv(x)
            else:
                x = self.act(conv(x))
            feats.append(x)
        score = x.view(x.shape[0], -1)
        return score, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(spectral_norm),
                ScaleDiscriminator(weight_norm),
                ScaleDiscriminator(weight_norm),
            ]
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        scores = []
        feats_all = []
        cur = x
        for i, d in enumerate(self.discriminators):
            if i > 0:
                cur = self.pool(cur)
            s, f = d(cur)
            scores.append(s)
            feats_all.append(f)
        return scores, feats_all


class HiFiGAN(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.generator = HiFiGANGenerator(in_channels=n_mels)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, **batch):
        if "audio" in batch:
            if len(batch["audio"].shape) < 3:
                batch["audio"] = batch["audio"].unsqueeze(1)

            batch["real_scores_mpd"], batch["real_feats_mpd"] = self.mpd(batch["audio"])
            batch["real_scores_msd"], batch["real_feats_msd"] = self.msd(batch["audio"])

        batch["audio_pred"] = self.generator(batch["mel"])
        batch["audio_pred_detached"] = batch["audio_pred"].clone().detach()

        batch["fake_scores_mpd"], batch["fake_feats_mpd"] = self.mpd(
            batch["audio_pred"]
        )
        batch["fake_scores_msd"], batch["fake_feats_msd"] = self.msd(
            batch["audio_pred"]
        )

        batch["fake_scores_mpd_detached"], batch["fake_feats_mpd_detached"] = self.mpd(
            batch["audio_pred_detached"]
        )
        batch["fake_scores_msd_detached"], batch["fake_feats_msd_detached"] = self.msd(
            batch["audio_pred_detached"]
        )

        return batch
