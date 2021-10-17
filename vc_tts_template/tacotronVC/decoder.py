# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

import torch
from torch import nn
from vc_tts_template.tacotron.attention import LocationSensitiveAttention

sys.path.append("..")
from vc_tts_template.utils import make_pad_mask
from vc_tts_template.tacotron.decoder import decoder_init, ZoneOutCell, Prenet


class Decoder(nn.Module):
    """Decoder of Tacotron 2.

    Args:
        encoder_hidden_dim (int) : dimension of encoder hidden layer
        out_dim (int) : dimension of output
        layers (int) : number of LSTM layers
        hidden_dim (int) : dimension of hidden layer
        prenet_layers (int) : number of pre-net layers
        prenet_hidden_dim (int) : dimension of pre-net hidden layer
        prenet_dropout (float) : dropout rate of pre-net
        zoneout (float) : zoneout rate
        reduction_factor (int) : reduction factor
        attention_hidden_dim (int) : dimension of attention hidden layer
        attention_conv_channel (int) : number of attention convolution channels
        attention_conv_kernel_size (int) : kernel size of attention convolution
    """

    def __init__(
        self,
        encoder_hidden_dim=512,  # エンコーダの隠れ層の次元数
        out_dim=80,  # 出力の次元数
        layers=2,  # LSTM 層の数
        hidden_dim=1024,  # LSTM層の次元数
        prenet_layers=2,  # Pre-Net の層の数
        prenet_hidden_dim=256,  # Pre-Net の隠れ層の次元数
        prenet_dropout=0.5,  # Pre-Net の Dropout 率
        zoneout=0.1,  # Zoneout 率
        reduction_factor=1,  # Reduction factor
        attention_hidden_dim=128,  # アテンション層の次元数
        attention_conv_channels=32,  # アテンションの畳み込みのチャネル数
        attention_conv_kernel_size=31,  # アテンションの畳み込みのカーネルサイズ
    ):
        super().__init__()
        self.out_dim = out_dim

        # 注意機構
        self.attention = LocationSensitiveAttention(
            encoder_hidden_dim,
            hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.reduction_factor = reduction_factor

        # Pre-Net
        self.prenet = Prenet(out_dim, prenet_layers, prenet_hidden_dim, prenet_dropout)

        # 片方向 LSTM
        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                encoder_hidden_dim + prenet_hidden_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        # 出力への projection 層
        proj_in_dim = encoder_hidden_dim + hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)
        self.prob_out = nn.Linear(proj_in_dim, reduction_factor)

        self.apply(decoder_init)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, encoder_outs, in_lens, decoder_targets=None, is_inference=False):
        """
        if (decoder_targets is None):
            do not use teacher forcing.
            do not use target for get same length.
        elif (decoder_targets is not None) and (is_inference is False):
            use teacher forcing.
            use target for get same length.
        elif (decoder_targets is not None) and (is_inference is True):
            do not use teacher forcing.
            use target for get same length.
        """
        # Reduction factor に基づくフレーム数の調整
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if (self.reduction_factor > 1) and (decoder_targets is not None):
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1:: self.reduction_factor
            ]

        # デコーダの系列長を保持
        # 推論時は、エンコーダの系列長から経験的に上限を定める
        if decoder_targets is None:
            max_decoder_time_steps = int(encoder_outs.shape[1] * 1.5)
        else:
            max_decoder_time_steps = decoder_targets.shape[1]

        # ゼロパディングされた部分に対するマスク
        mask = make_pad_mask(in_lens).to(encoder_outs.device)

        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        # デコーダの最初の入力
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        # 1 つ前の時刻のアテンション重み
        prev_att_w = None
        self.attention.reset()

        # メインループ
        outs, logits, att_ws = [], [], []
        t = 0
        while True:
            # コンテキストベクトル、アテンション重みの計算
            att_c, att_w = self.attention(
                encoder_outs, in_lens, h_list[0], prev_att_w, mask
            )

            # Pre-Net
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([att_c, prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # 出力の計算
            hcs = torch.cat([h_list[-1], att_c], dim=1)
            outs.append(self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1))
            logits.append(self.prob_out(hcs))
            att_ws.append(att_w)

            # 次の時刻のデコーダの入力を更新
            if is_inference is True:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

            # 累積アテンション重み
            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
            # 停止条件のチェック
            if t >= max_decoder_time_steps:
                break
            if (decoder_targets is None) and (is_inference is True) and (torch.sigmoid(logits[-1]) >= 0.5).any():
                break

        # 各時刻の出力を結合
        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.out_dim, -1)  # (B, out_dim, Lmax)

        return outs, logits, att_ws
