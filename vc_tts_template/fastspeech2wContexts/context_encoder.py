# https://github.dev/keonlee9420/Expressive-FastSpeech2/tree/conversational
import sys

import torch
import torch.nn as nn
import numpy as np

sys.path.append("../..")
from vc_tts_template.fastspeech2wGMM.layers import GRUwSort
from vc_tts_template.utils import make_pad_mask


class ConversationalContextEncoder(nn.Module):
    """ Conversational Context Encoder """

    def __init__(
        self,
        d_encoder_hidden,
        d_context_hidden,
        context_layer_num,
        context_dropout,
        text_emb_size,
    ):
        super(ConversationalContextEncoder, self).__init__()
        d_model = d_encoder_hidden
        d_cont_enc = d_context_hidden
        num_layers = context_layer_num
        dropout = context_dropout
        self.text_emb_size = text_emb_size

        self.text_emb_linear = nn.Linear(self.text_emb_size, d_cont_enc)
        self.speaker_linear = nn.Linear(d_model, d_cont_enc)

        self.enc_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )
        self.gru = GRUwSort(
            input_size=d_cont_enc,
            hidden_size=d_cont_enc,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
            sort=True,
        )
        self.gru_linear = nn.Sequential(
            nn.Linear(2*d_cont_enc, d_cont_enc),
            nn.ReLU()
        )

        self.context_linear = nn.Linear(d_cont_enc, d_model)
        self.context_attention = SLA(d_model)

    def forward(self, text_emb, speaker, history_text_emb, history_speaker, history_lens):
        max_history_len = history_text_emb.size(1)

        history_masks = make_pad_mask(history_lens, max_history_len)

        # Embedding
        history_text_emb = torch.cat([history_text_emb, text_emb.unsqueeze(1)], dim=1)
        history_text_emb = self.text_emb_linear(history_text_emb)
        history_speaker = torch.cat([history_speaker, speaker.unsqueeze(1)], dim=1)
        history_speaker = self.speaker_linear(history_speaker)

        history_enc = torch.cat([history_text_emb, history_speaker], dim=-1)
        history_enc = self.enc_linear(history_enc)

        # Split
        enc_past, enc_current = torch.split(history_enc, max_history_len, dim=1)

        # GRU
        enc_past = self.gru_linear(self.gru(enc_past, history_lens)[0])
        enc_past = enc_past.masked_fill(history_masks.unsqueeze(-1), 0)

        # Encoding
        context_enc = torch.cat([enc_current, enc_past], dim=1)
        context_enc = self.context_attention(self.context_linear(context_enc))  # [B, d]

        return context_enc


class SLA(nn.Module):
    """ Sequence Level Attention """

    def __init__(self, d_enc):
        super(SLA, self).__init__()
        self.linear = nn.Linear(d_enc, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoding, mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            encoding = encoding.to(torch.float32)
            attn = self.linear(encoding)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1), -np.inf)
            aux_mask = (attn == -np.inf).all(self.softmax.dim).unsqueeze(self.softmax.dim)
            attn = attn.masked_fill(aux_mask, 0)  # Remove all -inf along softmax.dim
        score = self.softmax(attn).transpose(-2, -1)  # [B, 1, T]
        fused_rep = torch.matmul(score, encoding).squeeze(1)  # [B, d]

        return fused_rep
