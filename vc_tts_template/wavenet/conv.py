import torch
from packaging import version
from torch import nn
from torch.nn import functional as F

torch_is_ge_180 = version.parse(torch.__version__) >= version.parse("1.8.0")


class Conv1d(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        if torch_is_ge_180:
            self.register_full_backward_hook(self._clear_linearized_weight)
        else:
            self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        # bufferが可能な, convを1から書いている. 難しい.
        if self.training:
            raise RuntimeError("incremental_forward only supports eval mode")

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(
                    bsz, kw + (kw - 1) * (dilation - 1), input.size(2)
                )
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        with torch.no_grad():
            output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        # 2次元のweightにする. これによって畳み込み計算ができる.
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            assert self.weight.size() == (self.out_channels, self.in_channels, kw)
            weight = self.weight.transpose(1, 2).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None
