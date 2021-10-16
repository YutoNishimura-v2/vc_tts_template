import torch
from torch import nn


from vc_tts_template.tacotronVC.decoder import Decoder
from vc_tts_template.tacotronVC.encoder import Encoder
from vc_tts_template.tacotron.postnet import Postnet
from vc_tts_template.utils import make_pad_mask
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
        n_mel_channel,
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
    ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.mel_num = n_mel_channel

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

    def init_forward(
        self,
        s_mels,
        s_mel_lens,
        max_s_mel_len,
        t_mel_lens,
        max_t_mel_len,
    ):
        t_mel_masks = (
            make_pad_mask(t_mel_lens, max_t_mel_len)
            if t_mel_lens is not None
            else None
        )
        if self.reduction_factor > 1:
            max_s_mel_len = max_s_mel_len // self.reduction_factor
            s_mel_lens = torch.trunc(s_mel_lens / self.reduction_factor)
            s_mels = s_mels[:, :max_s_mel_len*self.reduction_factor, :]

        return (
            s_mels,
            s_mel_lens,
            t_mel_masks,
        )

    def encoder_forward(
        self,
        s_sp_ids,
        s_em_ids,
        s_mels,
        s_mel_lens,
    ):
        output = self.mel_linear_1(
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
    ):
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(t_sp_ids).unsqueeze(1).expand(-1, output.size(1), -1)
        if self.emotion_emb is not None:
            output = output + self.speaker_emb(t_em_ids).unsqueeze(1).expand(-1, output.size(1), -1)

        output, logits, att_ws = self.decoder(output, s_mel_lens, t_mels)

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
        s_mels, s_mel_lens, t_mel_masks = self.init_forward(
            s_mels, s_mel_lens, max_s_mel_len, t_mel_lens, max_t_mel_len
        )
        output = self.encoder_forward(
            s_sp_ids, s_em_ids, s_mels, s_mel_lens,
        )

        # デコーダによるメルスペクトログラム、stop token の予測
        outs, outs_fine, logits, att_ws = self.decoder_forward(
            output, t_sp_ids, t_em_ids, s_mel_lens, t_mels
        )

        return outs, outs_fine, logits, att_ws, t_mel_masks


if __name__ == '__main__':
    # dummy test
    from frontend.text import text_to_sequence
    from vc_tts_template.utils import pad_1d

    model = Tacotron2VC(encoder_conv_layers=1, decoder_prenet_layers=1, decoder_layers=1,
                        postnet_layers=1, reduction_factor=3)

    def get_dummy_input():
        # バッチサイズに 2 を想定して、適当な文字列を作成
        seqs = [
            text_to_sequence("What is your favorite language?"),
            text_to_sequence("Hello world."),
        ]
        in_lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        max_len = max(len(x) for x in seqs)
        seqs = torch.stack([torch.from_numpy(pad_1d(seq, max_len)) for seq in seqs])

        return seqs, in_lens

    def get_dummy_inout():
        seqs, in_lens = get_dummy_input()

        # デコーダの出力（メルスペクトログラム）の教師データ
        decoder_targets = torch.ones(2, 120, 80)

        # stop token の教師データ
        # stop token の予測値は確率ですが、教師データは 二値のラベルです
        # 1 は、デコーダの出力が完了したことを表します
        stop_tokens = torch.zeros(2, 120)
        stop_tokens[:, -1:] = 1.0

        return seqs, in_lens, decoder_targets, stop_tokens

    # 適当な入出力を生成
    seqs, in_lens, decoder_targets, stop_tokens = get_dummy_inout()

    # Tacotron 2 の出力を計算
    # NOTE: teacher-forcing のため、 decoder targets を明示的に与える
    outs, outs_fine, logits, att_ws = model(seqs, in_lens, decoder_targets)

    print("入力のサイズ:", tuple(seqs.shape))
    print("デコーダの出力のサイズ:", tuple(outs.shape))
    print("Stop token のサイズ:", tuple(logits.shape))
    print("アテンション重みのサイズ:", tuple(att_ws.shape))
