import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
from torch.autograd import Variable


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


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


class GRUwSort(nn.Module):
    """
    pack_padded_sequenceなどを内部でやってくれるクラスです.
    """
    def __init__(self, input_size, hidden_size, num_layers,
                 batch_first, bidirectional, sort, dropout=0.0,
                 allow_zero_length=False, need_last=False) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout
        )
        self.sort = sort
        self.batch_first = batch_first
        self.zero_num = 0
        self.allow_zero_length = allow_zero_length
        self.need_last = need_last

        self.output_dim = hidden_size * 2 if bidirectional is True else hidden_size

    def forward(self, x, lens):
        if self.sort is True:
            if type(lens) == torch.Tensor:
                sort_idx = torch.argsort(-lens)
                inv_sort_idx = torch.argsort(sort_idx)
            else:
                lens = np.array(lens)
                sort_idx = np.argsort(-lens)
                inv_sort_idx = np.argsort(sort_idx)
            x = x[sort_idx]
            lens = lens[sort_idx]

        if type(lens) == torch.Tensor:
            lens = lens.to("cpu").numpy()

        if self.allow_zero_length is True:
            x, lens = self.remove_zeros(x, lens)

        x = pack_padded_sequence(x, lens, batch_first=self.batch_first)
        out, _ = self.gru(x)
        out, _ = pad_packed_sequence(out, batch_first=self.batch_first)

        if self.allow_zero_length is True:
            out = self.restore_zeros(out)

        if self.need_last is True:
            out = self.get_last_timestep(out, lens)

        if self.sort is True:
            out = out[inv_sort_idx]

        return out

    def remove_zeros(self, x, lens):
        # 最新実装ではもはやremoveしていないことに注意
        self.zero_num = np.sum(lens == 0)
        if self.zero_num == 0:
            return x, lens
        # 1にしておけば, 最後のget_last_timestepでも実装が容易になる
        lens[-self.zero_num:] = 1

        if x.size(1) == 0:
            raise ValueError("未対応です. xのtimeは少なくとも1以上にしてください.")

        return x, lens

    def restore_zeros(self, x):
        if self.zero_num > 0:
            x[-self.zero_num:] = 0.0
        return x

    def get_last_timestep(self, out, lens):
        idx = (torch.LongTensor(lens) - 1).view(-1, 1).expand(
            len(lens), out.size(2))
        time_dimension = 1 if self.batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(out.device)
        out = out.gather(
            time_dimension, Variable(idx)
        ).squeeze(time_dimension)
        return out
