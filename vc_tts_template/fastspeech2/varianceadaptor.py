import sys
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
from vc_tts_template.utils import make_pad_mask, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
        pitch_feature_level: int,
        energy_feature_level: int,
        pitch_quantization: str,
        energy_quantization: str,
        n_bins: int,
        stats: Dict
    ):
        super(VarianceAdaptor, self).__init__()
        # duration, pitch, energyで共通なのね.
        self.duration_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )
        self.energy_predictor = VariancePredictor(
            encoder_hidden_dim,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
        )

        # default: pitch: feature: "phoneme_level"
        self.pitch_feature_level = "phoneme_level" if pitch_feature_level > 0 else "frame_level"
        self.energy_feature_level = "phoneme_level" if energy_feature_level > 0 else "frame_level"

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        # default: variance_embedding: pitch_quantization: "linear"
        pitch_quantization = pitch_quantization
        energy_quantization = energy_quantization
        # default: n_bins: 256
        n_bins = n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]

        pitch_min, pitch_max = stats["pitch_min"], stats["pitch_max"]
        energy_min, energy_max = stats["energy_min"], stats["energy_max"]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min),
                                   np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min),
                                   np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        # pitchとenergyに関してはなんとembedding!
        self.pitch_embedding = nn.Embedding(
            n_bins, encoder_hidden_dim
        )
        self.energy_embedding = nn.Embedding(
            n_bins, encoder_hidden_dim
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:  # つまりtrain時.
            # bucketizeで, その値がどのself.pitch_binsの間に入っているかを調べて, そのindexを返す.
            # 例: target = 2, pitch_bins = 1,3,5
            # なら, returnは, 1となる.
            # 要するに, pitchの大きさに対する特徴を学習させようとしている.
            # embeddingで次元を膨らませている感じ(既に, targetだけでpitchの値はあるにはあるので)
            embedding = self.pitch_embedding(
                torch.bucketize(target, self.pitch_bins))
        else:  # inference時.
            # 同様に, こっちはpredictionに対して. それはそうだが.
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        # pitchとやっていることは同じ.
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(
                torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # まずは, durationを計算する.
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # pitchがphoneme, つまりframe_levelではない場合
        if self.pitch_feature_level == "phoneme_level":
            # get_pitch_embeddingはこのクラスの関数.
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding

        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        # durationの正解データがあるのであれば, targetとともにreguratorへ.
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            # そうでないなら, predictionを利用.
            duration_rounded = torch.clamp(  # 最小値を0にする. マイナスは許さない.
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            # そして, predictで作ったduration_roundedを使ってregulatorへ.
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            # inferenceではmel_maskもないので, Noneとしてくる.
            mel_mask = make_pad_mask(mel_len)

        if self.pitch_feature_level == "frame_level":
            # frame_levelなら, 一気に見るので, src_maskではなく, mel_maskを見てもらう.
            # mel_maskじゃないと次元も合わないよね.
            # 違いはそこだけ.
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            # embbeddingのほうを足しておく.
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            # expandedのlenがまさに出力したいmelの幅になる.
            mel_len.append(expanded.shape[0])

        # ここでは, まだoutputは長さバラバラのlistであることに注意.
        # 長さを揃えなきゃ.
        if max_len is not None:
            # max_lenがあるなら, それでpad.
            output = pad(output, max_len)
        else:
            # targetがないならmax_lenもないですね.
            # その場合は自動で一番長い部分を探してくれる.
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)  # listをtorchの2次元へ, かな.

        return out

    def forward(self, x, duration, max_len):
        # durationがpredictか, targetのdurationです.
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim: int,
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
    ):
        super(VariancePredictor, self).__init__()

        self.input_size = encoder_hidden_dim
        self.filter_size = variance_predictor_filter_size
        self.kernel = variance_predictor_kernel_size
        self.conv_output_size = variance_predictor_filter_size
        self.dropout = variance_predictor_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(  # なんでわざわざ? 命名のためかな?
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,  # same sizeに.
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
