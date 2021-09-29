import torch.nn as nn

from vc_tts_template.fastspeech2VC.modules import Transpose


class ConvBNorms2d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        n_layers=2,
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.convolutions = []

        self.convolutions += [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

        for _ in range(n_layers - 1):
            self.convolutions += [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
        self.convolutions = nn.Sequential(*self.convolutions)

    def forward(self, x):
        # expected (B, C, H, W)
        return self.convolutions(x)


class ConvLNorms1d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        n_layers=2,
        drop_out=0.2
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.convolutions = []

        self.convolutions += [
            Transpose(shape=(1, 2)),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.ReLU(),
            Transpose(shape=(1, 2)),
            nn.LayerNorm(out_channels),
            nn.Dropout(drop_out)
        ]

        for _ in range(n_layers - 1):
            self.convolutions += [
                Transpose(shape=(1, 2)),
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias
                ),
                nn.ReLU(),
                Transpose(shape=(1, 2)),
                nn.LayerNorm(out_channels),
                nn.Dropout(drop_out)
            ]
        self.convolutions = nn.Sequential(*self.convolutions)

    def forward(self, x):
        # expected (B, T, d)
        return self.convolutions(x)
