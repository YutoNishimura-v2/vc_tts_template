import numpy as np
from torch import nn
from torch.nn import functional as F

from .modules import Conv1d  # moduleの.

__all__ = [
    "RepeatUpsampling",
    "ConvTransposeUpsampleNetwork",
    "UpsampleNetwork",
    "ConvInUpsampleNetwork",
]


class RepeatUpsampling(nn.Module):
    """Repeat upsampling
    一番シンプルなやつ.

    Args:
        upsample_scales (list): list of scales to upsample
    """

    def __init__(self, upsample_scales):
        super().__init__()
        # np.prodは, 積のPiのやつ.
        # 後半での実装を見据えた引数になっている.
        self.upsample = nn.Upsample(scale_factor=np.prod(upsample_scales))

    def forward(self, c):
        """Forward step

        Args:
            c (torch.Tensor): input features

        Returns:
            torch.Tensor: upsampled features
        """
        return self.upsample(c)


class UpsampleNetwork(nn.Module):
    """Upsample by nearest neighbor

    Args:
        upsample_scales (list): list of scales to upsample
    """

    def __init__(self, upsample_scales):
        super().__init__()
        self.upsample_scales = upsample_scales
        self.conv_layers = nn.ModuleList()
        for scale in upsample_scales:
            kernel_size = (1, scale * 2 + 1)
            conv = nn.Conv2d(
                1, 1, kernel_size=kernel_size, padding=(0, scale), bias=False
            )
            conv.weight.data.fill_(1.0 / np.prod(kernel_size))  # これで初期化することにより, 最初はただのidentityになる.
            self.conv_layers.append(nn.utils.weight_norm(conv))  # こっそりweight_normを入れることを忘れずに.
            # ↑しかもweight norm入ってるから常にスケールはキープされる. 便利.

    def forward(self, c):
        """Forward step

        Args:
            c (torch.Tensor): input features

        Returns:
            torch.Tensor: upsampled features
        """
        # (B, 1, C, T)
        c = c.unsqueeze(1)
        # 最近傍補完と畳み込みと畳み込みの繰り返し
        for idx, scale in enumerate(self.upsample_scales):
            # 時間方向にのみアップサンプリング
            # (B, 1, C, T) -> (B, 1, C, T*scale)
            # 単純にコピーして増やして, その後いい感じにlinear入れる感じ.
            c = F.interpolate(c, scale_factor=(1, scale), mode="nearest")
            c = self.conv_layers[idx](c)
        # B x C x T
        return c.squeeze(1)


class ConvInUpsampleNetwork(nn.Module):
    """Conv1d + UpsampleNetwork
    upsampleするまえに, 近傍情報を入れておく.

    Args:
        upsample_scales (list): list of scales to upsample
        cin_channels (int): number of input channels
        aux_context_window (int): size of the auxiliary context window
    """

    def __init__(self, upsample_scales, cin_channels, aux_context_window):
        super().__init__()
        # 条件付け特徴量近傍を、1 次元畳み込みによって考慮します
        kernel_size = 2 * aux_context_window + 1
        self.conv_in = Conv1d(cin_channels, cin_channels, kernel_size, bias=False)
        self.upsample = UpsampleNetwork(upsample_scales)

    def forward(self, c):
        """Forward step

        Args:
            c (torch.Tensor): input features

        Returns:
            torch.Tensor: upsampled features
        """
        return self.upsample(self.conv_in(c))


class ConvTransposeUpsampleNetwork(nn.Module):  # 未使用...。
    """Upsampling based on transposed convolution
    転置畳み込みのみによるupsampling. 本において, ゆがみが生じるとしてスルーされた手法.

    Args:
        upsample_scales (list): list of scales to upsample
        aux_context_window (int): size of the auxiliary context window
    """

    def __init__(self, upsample_scales, aux_context_window):
        super().__init__()
        self.up_layers = nn.ModuleList()
        self.upsample_scales = upsample_scales
        total_scale = np.prod(upsample_scales)

        for scale in upsample_scales:
            kernel_size = (1, 2 * scale)
            convt = nn.ConvTranspose2d(
                1,
                1,
                kernel_size,
                padding=(0, scale // 2),
                dilation=1,
                stride=(1, scale),
            )
            convt.weight.data.fill_(0.5)
            convt.bias.data.fill_(0)
            self.up_layers.append(convt)
        self.trim_length = aux_context_window * total_scale

    def forward(self, c):
        """Forward step

        Args:
            c (torch.Tensor): input features

        Returns:
            torch.Tensor: upsampled features
        """
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        c = c.squeeze(1)
        if self.trim_length > 0:
            c = c[:, :, self.trim_length: -self.trim_length]
        return c
