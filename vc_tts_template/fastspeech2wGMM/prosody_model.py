import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append("../..")
from vc_tts_template.utils import pad


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))


class ProsodyExtractor(nn.Module):
    def __init__(
        self,
        d_mel=80,
        d_out=128,
        conv_in_channels=1,
        conv_out_channels=1,
        conv_kernel_size=1,
        conv_stride=1,
        conv_padding=None,
        conv_dilation=1,
        conv_bias=True,
        conv_n_layers=2,
    ):
        super().__init__()
        self.convnorms = ConvNorms(
            conv_in_channels, conv_out_channels, conv_kernel_size,
            conv_stride, conv_padding, conv_dilation, conv_bias, conv_n_layers
        )
        self.convnorms.apply(encoder_init)
        self.bi_gru = nn.GRU(
            input_size=d_mel, hidden_size=d_out // 2, num_layers=1,
            batch_first=True, bidirectional=True
        )

    def forward(self, mels, durations):
        # expected(B, T, d_mel)
        durations = durations.detach().cpu().numpy()  # 念のため, 伝播を防ぐ.
        # melをphone単位のsegmentに分割
        output_sorted, src_lens_sorted, segment_nums, inv_sort_idx = self.mel2phone(mels, durations)
        outs = list()
        for output, src_lens in zip(output_sorted, src_lens_sorted):
            # convolution
            mel_hidden = self.convnorms(output.unsqueeze(1))
            # Bi-GRU
            mel_hidden = pack_padded_sequence(mel_hidden.squeeze(1), src_lens, batch_first=True)
            out, _ = self.bi_gru(mel_hidden)
            out, _ = pad_packed_sequence(out, batch_first=True)
            outs.append(out[:, -1, :])  # 最終時刻の出力を利用
        outs = torch.cat(outs, 0)
        out = self.phone2emb(outs[inv_sort_idx], segment_nums)
        return out

    def mel2phone(self, mels, durations):
        output = list()
        output_sorted = list()
        src_lens = list()
        src_lens_sorted = list()
        segment_nums = list()
        for mel, duration in zip(mels, durations):
            s_idx = 0
            cnt = 0
            for d in duration:
                if d == 0:
                    continue
                mel_seg = mel[s_idx: s_idx+d]
                s_idx += d
                cnt += 1
                output.append(mel_seg)
                src_lens.append(d)
            segment_nums.append(cnt)

        sort_idx = np.argsort(-np.array(src_lens))
        # 同じ値があっても元の場所に戻るのは確認済
        inv_sort_idx = np.argsort(sort_idx)
        # padを少なくするために, batch数ごとのまとまりでpadしていく
        # 以下のやり方は, BNをする際にbatch_size=1とならないようにするための方法
        batch_size = mels.size(0)
        for i in range(len(output) // batch_size):
            sort_seg = sort_idx[i*batch_size:(i+1)*batch_size]
            if i == ((len(output) // batch_size) - 1):
                sort_seg = sort_idx[i*batch_size:]
            output_seg = [output[idx] for idx in sort_seg]
            src_lens_seg = [src_lens[idx] for idx in sort_seg]
            output_seg = pad(output_seg)
            output_sorted.append(output_seg)
            src_lens_sorted.append(src_lens_seg)
        return output_sorted, src_lens_sorted, segment_nums, inv_sort_idx

    def phone2emb(self, out, segment_nums):
        output = list()
        s_idx = 0
        for seg_num in segment_nums:
            output.append(out[s_idx:s_idx+seg_num])
            s_idx += seg_num
        return pad(output)


class ConvNorms(nn.Module):
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

        self.convolutions.append(
            nn.Sequential(
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
            )
        )

        for _ in range(1, n_layers - 1):
            self.convolutions.append(
                nn.Sequential(
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
                )
            )
        self.convolutions = nn.Sequential(*self.convolutions)

    def forward(self, x):
        # expected (B, C, H, W)
        return self.convolutions(x)


if __name__ == "__main__":
    """
    以下のmelとdurationでは, mel_segments[0]は
    [1,2,0,0,0,0,0,0,0,0]になる.
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ProsodyExtractor(d_mel=1, d_out=4).to(device)
    mels = torch.Tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ]
    ).unsqueeze(-1)  # (B, T, mel_bin)
    durations = torch.Tensor(
        [
            [2, 2, 2, 2, 2],
            [2, 3, 2, 0, 0],
            [10, 0, 0, 0, 0]
        ]
    ).long()
    out = extractor(mels.to(device), durations.to(device))
    print(out.size())