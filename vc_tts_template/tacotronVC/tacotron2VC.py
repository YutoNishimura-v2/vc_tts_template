import sys
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

sys.path.append("../..")
from vc_tts_template.tacotronVC.decoder import Decoder
from vc_tts_template.tacotronVC.encoder import Encoder
from vc_tts_template.tacotron.postnet import Postnet
from vc_tts_template.utils import make_pad_mask, pad
from vc_tts_template.train_utils import free_tensors_memory


class Tacotron2VC(nn.Module):
    """Tacotron 2VC

    This implementation does not include the WaveNet vocoder of the Tacotron 2.

    Args:
        num_vocab (int): the size of vocabulary
        embed_dim (int): dimension of embedding
        encoder_hidden_dim (int): dimension of hidden unit
        encoder_conv_layers (int): the number of convolution layers
        encoder_conv_channels (int): the number of convolution channels
        encoder_conv_kernel_size (int): kernel size of convolution
        encoder_dropout (float): dropout rate of convolution
        attention_hidden_dim (int): dimension of hidden unit
        attention_conv_channels (int): the number of convolution channels
        attention_conv_kernel_size (int): kernel size of convolution
        decoder_out_dim (int): dimension of output
        decoder_layers (int): the number of decoder layers
        decoder_hidden_dim (int): dimension of hidden unit
        decoder_prenet_layers (int): the number of prenet layers
        decoder_prenet_hidden_dim (int): dimension of hidden unit
        decoder_prenet_dropout (float): dropout rate of prenet
        decoder_zoneout (float): zoneout rate
        postnet_layers (int): the number of postnet layers
        postnet_channels (int): the number of postnet channels
        postnet_kernel_size (int): kernel size of postnet
        postnet_dropout (float): dropout rate of postnet
        reduction_factor (int): reduction factor
    """

    def __init__(
        self,
        n_mel_channel=80,
        # encoder
        encoder_hidden_dim=512,
        encoder_conv_layers=3,
        encoder_conv_channels=512,
        encoder_conv_kernel_size=5,
        encoder_dropout=0.5,
        # attention
        attention_hidden_dim=128,
        attention_conv_channels=32,
        attention_conv_kernel_size=31,
        # decoder
        decoder_layers=2,
        decoder_hidden_dim=1024,
        decoder_prenet_layers=2,
        decoder_prenet_hidden_dim=256,
        decoder_prenet_dropout=0.5,
        decoder_zoneout=0.1,
        postnet_layers=5,
        postnet_channels=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,
        reduction_factor=1,
        # other
        encoder_fix: bool = False,
        speakers: Optional[Dict] = None,
        emotions: Optional[Dict] = None
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.mel_num = n_mel_channel

        self.mel_linear = nn.Linear(
            self.mel_num * self.reduction_factor,
            encoder_hidden_dim,
        )
        self.encoder = Encoder(
            encoder_hidden_dim,
            encoder_conv_layers,
            encoder_conv_channels,
            encoder_conv_kernel_size,
            encoder_dropout,
        )
        self.decoder = Decoder(
            encoder_hidden_dim,
            self.mel_num,
            decoder_layers,
            decoder_hidden_dim,
            decoder_prenet_layers,
            decoder_prenet_hidden_dim,
            decoder_prenet_dropout,
            decoder_zoneout,
            reduction_factor,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.postnet = Postnet(
            self.mel_num,
            postnet_layers,
            postnet_channels,
            postnet_kernel_size,
            postnet_dropout,
        )
        self.speaker_emb = None
        if speakers is not None:
            n_speaker = len(speakers)
            self.speaker_emb = nn.Embedding(
                n_speaker,
                encoder_hidden_dim,
            )
        self.emotion_emb = None
        if emotions is not None:
            n_emotion = len(emotions)
            self.emotion_emb = nn.Embedding(
                n_emotion,
                encoder_hidden_dim,
            )
        self.encoder_fix = encoder_fix
        self.speakers = speakers
        self.emotions = emotions

    def init_forward(
        self,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mel_lens,
        max_t_mel_len,
    ):
        if self.reduction_factor > 1:
            max_s_mel_len = max_s_mel_len // self.reduction_factor
            s_mel_lens = torch.trunc(s_mel_lens / self.reduction_factor)
            s_mels = s_mels[:, :max_s_mel_len*self.reduction_factor, :]

            if t_mel_lens is not None:
                t_mel_lens = torch.trunc(t_mel_lens / self.reduction_factor) * self.reduction_factor
                max_t_mel_len = max_t_mel_len // self.reduction_factor * self.reduction_factor

        t_mel_masks = (
            make_pad_mask(t_mel_lens, max_t_mel_len)
            if t_mel_lens is not None
            else None
        )

        return (
            s_mels,
            s_mel_lens,
            t_mel_lens,
            t_mel_masks,
        )

    def encoder_forward(
        self,
        s_sp_ids,
        s_em_ids,
        s_mels,
        s_mel_lens,
    ):
        output = self.mel_linear(
            s_mels.contiguous().view(s_mels.size(0), -1, self.mel_num * self.reduction_factor)
        )
        free_tensors_memory([s_mels])

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(s_sp_ids).unsqueeze(1).expand(-1, output.size(1), -1)
        if self.emotion_emb is not None:
            output = output + self.emotion_emb(s_em_ids).unsqueeze(1).expand(-1, output.size(1), -1)

        output = self.encoder(output, s_mel_lens)

        if self.encoder_fix is True:
            output = output.detach()

        return output

    def decoder_forward(
        self,
        output,
        t_sp_ids,
        t_em_ids,
        s_mel_lens,
        t_mels,
        is_inference=False
    ):
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(t_sp_ids).unsqueeze(1).expand(-1, output.size(1), -1)
        if self.emotion_emb is not None:
            output = output + self.emotion_emb(t_em_ids).unsqueeze(1).expand(-1, output.size(1), -1)

        output, logits, att_ws = self.decoder(output, s_mel_lens, t_mels, is_inference)

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = output + self.postnet(output)

        # (B, C, T) -> (B, T, C)
        output = output.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return output, outs_fine, logits, att_ws

    def forward(
        self,
        ids,
        s_sp_ids,
        t_sp_ids,
        s_em_ids,
        t_em_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
    ):
        s_mels, s_mel_lens, t_mel_lens, t_mel_masks = self.init_forward(
            s_mels, s_mel_lens, max_s_mel_len, t_mel_lens, max_t_mel_len
        )
        output = self.encoder_forward(
            s_sp_ids, s_em_ids, s_mels, s_mel_lens,
        )
        # デコーダによるメルスペクトログラム、stop token の予測
        outs, outs_fine, logits, att_ws = self.decoder_forward(
            output, t_sp_ids, t_em_ids, s_mel_lens, t_mels
        )

        return outs, outs_fine, logits, att_ws, t_mel_lens, t_mel_masks

    def inference(
        self,
        ids,
        s_sp_ids,
        t_sp_ids,
        s_em_ids,
        t_em_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
        max_synth_num=1000000,
    ):
        if t_mels is not None:
            # 主にteacher forcingなしでloss計算するvalid用.
            # mel_lenもあるのでbatch処理でよい.
            outs, outs_fine, logits, att_ws, t_mel_lens_pre, t_mel_masks = self._inference(
                ids,
                s_sp_ids,
                t_sp_ids,
                s_em_ids,
                t_em_ids,
                s_mels,
                s_mel_lens,
                max_s_mel_len,
                t_mels,
                t_mel_lens,
                max_t_mel_len,
            )
        else:
            # 主にsynthesis用.
            # 1つずつ処理する.
            outs = []
            outs_fine = []
            logits = []
            att_ws = []
            t_mel_lens_pre = []

            for idx in range(min(len(ids), max_synth_num)):
                out, out_fine, logit, att_w, _, _ = self._inference(
                    ids[idx],
                    s_sp_ids[idx].unsqueeze(0),
                    t_sp_ids[idx].unsqueeze(0),
                    s_em_ids[idx].unsqueeze(0),
                    t_em_ids[idx].unsqueeze(0),
                    s_mels[idx].unsqueeze(0),
                    s_mel_lens[idx].unsqueeze(0),
                    s_mel_lens[idx],
                )
                outs.append(out.squeeze(0))
                outs_fine.append(out_fine.squeeze(0))
                logits.append(logit.squeeze(0))
                att_ws.append(att_w.squeeze(0))
                t_mel_lens_pre.append(out_fine.size(1))

            t_mel_masks = make_pad_mask(t_mel_lens_pre, max(t_mel_lens_pre))
            t_mel_lens_pre = torch.from_numpy(np.array(t_mel_lens_pre)).long().to(t_mel_masks.device)
            outs = pad(outs)
            outs_fine = pad(outs_fine)
            logits = pad(logits)

        return outs, outs_fine, logits, att_ws, t_mel_lens_pre, t_mel_masks

    def _inference(
        self,
        ids,
        s_sp_ids,
        t_sp_ids,
        s_em_ids,
        t_em_ids,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
    ):
        s_mels, s_mel_lens, t_mel_lens, t_mel_masks = self.init_forward(
            s_mels, s_mel_lens, max_s_mel_len, t_mel_lens, max_t_mel_len
        )
        output = self.encoder_forward(
            s_sp_ids, s_em_ids, s_mels, s_mel_lens,
        )
        outs, outs_fine, logits, att_ws = self.decoder_forward(
            output, t_sp_ids, t_em_ids, s_mel_lens, t_mels, is_inference=True
        )
        return outs, outs_fine, logits, att_ws, t_mel_lens, t_mel_masks


if __name__ == '__main__':
    # dummy test
    from vc_tts_template.train_utils import plot_attention

    model = Tacotron2VC(encoder_conv_layers=1, decoder_prenet_layers=1, decoder_layers=1,
                        postnet_layers=1, reduction_factor=3)

    def get_dummy_input():
        # バッチサイズに 2 を想定.
        ids = torch.zeros(2)
        s_sp_ids = torch.zeros(2)
        t_sp_ids = torch.zeros(2)
        s_em_ids = torch.zeros(2)
        t_em_ids = torch.zeros(2)
        s_mels = torch.randn(2, 21, 80)
        s_mel_lens = torch.LongTensor([21, 10])
        max_s_mel_len = 21
        t_mels = torch.randn(2, 30, 80)
        t_mel_lens = torch.LongTensor([30, 15])
        max_t_mel_len = 30

        return (
            ids,
            s_sp_ids,
            t_sp_ids,
            s_em_ids,
            t_em_ids,
            s_mels,
            s_mel_lens,
            max_s_mel_len,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
        )

    # Tacotron 2 の出力を計算
    # NOTE: teacher-forcing のため、 decoder targets を明示的に与える
    outs, outs_fine, logits, att_ws, t_mel_masks = model(*get_dummy_input())

    print("デコーダの出力のサイズ:", tuple(outs.shape))
    print("Stop token のサイズ:", tuple(logits.shape))
    print("アテンション重みのサイズ:", tuple(att_ws.shape))

    fig = plot_attention(att_ws[0])
    attn_selected = att_ws[0]
    for idx, argmax_idx in enumerate(torch.argmax(attn_selected, dim=-1)):
        attn_selected[idx, :] = 0
        attn_selected[idx, argmax_idx] = 1

    fig2 = plot_attention(attn_selected)
    fig.savefig("tmp.png")
    fig2.savefig("tmp2.png")
