import sys

import torch
import torch.nn as nn

sys.path.append("../..")
from vc_tts_template.fastspeech2wGMM.prosody_model import ProsodyPredictor
from vc_tts_template.tacotron.decoder import LocationSensitiveAttention
from vc_tts_template.utils import make_pad_mask


class ProsodyPredictorwAttention(ProsodyPredictor):
    def __init__(
        self,
        d_in=1,
        d_gru=1,  # dimention of gru
        d_out=1,  # output prosody emb size
        conv_out_channels=1,  # hidden channel before gru
        conv_kernel_size=1,
        conv_stride=1,
        conv_padding=None,
        conv_dilation=1,
        conv_bias=True,
        conv_n_layers=2,
        conv_dropout=0.2,
        gru_layers=2,
        zoneout=0.1,
        num_gaussians=10,
        global_prosody=False,
        global_gru_layers=1,
        global_d_gru=256,
        global_num_gaussians=10,
        h_prosody_emb_size=256,
        attention_hidden_dim=512,
        attention_conv_channels=256,
        attention_conv_kernel_size=128,
    ) -> None:
        super().__init__(
            d_in, d_gru, d_out,
            conv_out_channels, conv_kernel_size, conv_stride,
            conv_padding, conv_dilation, conv_bias,
            conv_n_layers, conv_dropout, gru_layers,
            zoneout, num_gaussians, global_prosody,
            global_gru_layers, global_d_gru, global_num_gaussians,
        )
        self.attention = LocationSensitiveAttention(
            h_prosody_emb_size,
            d_gru,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.attn_linear = nn.Linear(h_prosody_emb_size, conv_out_channels+d_out)

    def forward(
        self, encoder_output, h_prosody_emb, h_prosody_lens,
        target_prosody=None, target_global_prosody=None,
        src_lens=None, src_mask=None, is_inference=False
    ):
        encoder_output = self.convnorms(encoder_output)

        if self.global_prosody is True:
            # hidden_global: (B, global_d_gru)
            hidden_global = self.global_bi_gru(encoder_output, src_lens)[:, -1, :]

            g_pi = self.g_pi_linear(hidden_global)
            g_sigma = (self.g_sigma_linear(hidden_global)+1.0).view(-1, self.global_num_gaussians, self.d_out)
            g_mu = self.g_mu_linear(hidden_global).view(-1, self.global_num_gaussians, self.d_out)
            if target_global_prosody is None:
                target_global_prosody = self.sample(g_pi, g_sigma, g_mu)
            else:
                target_global_prosody = target_global_prosody.detach()

        # GRU の状態をゼロで初期化
        h_list = []
        for _ in range(len(self.gru)):
            h_list.append(self._zero_state(encoder_output))

        # 最初の入力
        go_frame = encoder_output.new_zeros(encoder_output.size(0), self.d_out)
        prev_out = go_frame

        # 1 つ前の時刻のアテンション重み
        prev_att_w = None
        self.attention.reset()

        pi_outs = []
        sigma_outs = []
        mu_outs = []
        outs = []

        h_prosody_mask = make_pad_mask(h_prosody_lens)

        for t in range(encoder_output.size()[1]):
            # Pre-Net
            if target_global_prosody is not None:
                prev_out = prev_out + target_global_prosody
            prenet_out = self.prenet(prev_out)

            att_c, att_w = self.attention(
                h_prosody_emb, h_prosody_lens, h_list[0], prev_att_w, h_prosody_mask
            )

            # LSTM
            xs = torch.cat([encoder_output[:, t, :], prenet_out], dim=1)
            xs = xs + self.attn_linear(att_c)
            h_list[0] = self.gru[0](xs, h_list[0])
            for i in range(1, len(self.gru)):
                h_list[i] = self.gru[i](
                    h_list[i - 1], h_list[i]
                )
            hcs = torch.cat([h_list[-1], encoder_output[:, t, :]], dim=1)
            pi_outs.append(self.pi_linear(hcs).unsqueeze(1))
            sigma_outs.append((self.sigma_linear(hcs)+1.0).view(-1, 1, self.num_gaussians, self.d_out))
            mu_outs.append(self.mu_linear(hcs).view(-1, 1, self.num_gaussians, self.d_out))

            # 次の時刻のデコーダの入力を更新
            if (is_inference is True) or (target_prosody is None):
                prev_out = self.sample(
                    pi_outs[-1].squeeze(1), sigma_outs[-1].squeeze(1), mu_outs[-1].squeeze(1)
                )  # (B, d_out)
            else:
                # Teacher forcing
                # prevent from backpropagation to prosody extractor
                prev_out = target_prosody[:, t, :].detach()
            outs.append(prev_out.unsqueeze(1))

            # 累積アテンション重み
            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

        outs = torch.cat(outs, dim=1)
        pi_outs = torch.cat(pi_outs, dim=1)
        sigma_outs = torch.cat(sigma_outs, dim=1)
        mu_outs = torch.cat(mu_outs, dim=1)
        if self.global_prosody is True:
            return outs, pi_outs, sigma_outs, mu_outs, g_pi, g_sigma, g_mu
        return outs, pi_outs, sigma_outs, mu_outs
