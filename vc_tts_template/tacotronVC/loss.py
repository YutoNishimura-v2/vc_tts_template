import torch
import torch.nn as nn


def guided_attention(N, max_N, T, max_T, g, device):
    W = torch.zeros((max_N, max_T)).to(device)
    x = torch.arange(max_N).to(device) / N
    W = W + x.unsqueeze(1).expand(-1, W.size(1))
    y = torch.arange(max_T).to(device) / T
    W = W - y.unsqueeze(1).expand(-1, W.size(0)).T
    W = 1 - torch.exp(- (W * W) / (2 * g * g))
    W[N:max_N, :] = 0
    W[:, T:max_T] = 0

    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.2):
    # https://github.com/r9y9/deepvoice3_pytorch/blob/a5c24624bad314db5a5dcb0ea320fc3623a94f15/train.py#L594
    device = input_lengths.device
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    W = torch.zeros((B, max_target_len, max_input_len)).to(device)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g, device).T
    return W


class Tacotron2VCLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, guided_attention_sigma):
        super(Tacotron2VCLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.guided_attention_sigma = guided_attention_sigma

    def forward(self, inputs, predictions):
        (
            s_mel_lens,
            _,
            t_mels,
            t_mel_lens,
            max_t_mel_len,
        ) = inputs[-5:]
        (
            outs,
            outs_fine,
            logits,
            attn_ws,
            t_mel_lens,
            t_mel_masks
        ) = predictions

        t_mel_masks = ~t_mel_masks
        t_mels.requires_grad = False

        assert (t_mels.size(1) - outs.size(1)) < 3, f"{outs.size(1)}, {t_mels.size(1)}"
        assert (t_mels.size(1) - outs.size(1)) >= 0, f"{outs.size(1)}, {t_mels.size(1)}"
        t_mels = t_mels[:, :outs.size(1), :]

        # make stop flags
        stop_flags = torch.zeros(t_mels.size(0), t_mels.size(1)).to(t_mels.device)
        for idx, out_len in enumerate(t_mel_lens):
            stop_flags[idx, int(out_len) - 1:] = 1.0

        # make attn mask
        reduction_factor = int(max_t_mel_len / attn_ws.size(1))
        s_mel_lens = s_mel_lens / reduction_factor
        t_mel_lens = t_mel_lens / reduction_factor
        max_t_mel_len = max_t_mel_len / reduction_factor
        soft_mask = guided_attentions(
            input_lengths=s_mel_lens.long(),
            target_lengths=t_mel_lens.long(), max_target_len=int(max_t_mel_len),
            g=self.guided_attention_sigma
        )

        t_mels = t_mels.masked_select(t_mel_masks.unsqueeze(-1))
        outs = outs.masked_select(t_mel_masks.unsqueeze(-1))
        outs_fine = outs_fine.masked_select(t_mel_masks.unsqueeze(-1))
        stop_flags = stop_flags.masked_select(t_mel_masks)
        logits = logits.masked_select(t_mel_masks)

        # Loss
        decoder_out_loss = self.mse_loss(outs, t_mels)
        postnet_out_loss = self.mse_loss(outs_fine, t_mels)
        stop_token_loss = self.bce_loss(logits, stop_flags)
        attn_guided_loss = (attn_ws * soft_mask).mean()
        total_loss = decoder_out_loss + postnet_out_loss + stop_token_loss + attn_guided_loss

        loss_values = {
            "mel_loss": decoder_out_loss.item(),
            "postnet_mel_loss": postnet_out_loss.item(),
            "stoptoken_loss": stop_token_loss.item(),
            "attn_guided_loss": attn_guided_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_values
