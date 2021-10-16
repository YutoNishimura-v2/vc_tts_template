import torch
import torch.nn as nn


class Tacotron2VCLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(Tacotron2VCLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, predictions):
        (
            t_mels,
            t_mel_lens,
            _,
        ) = inputs[-3:]
        (
            outs,
            outs_fine,
            logits,
            _,
            t_mel_lens,
            t_mel_masks
        ) = predictions

        t_mel_masks = ~t_mel_masks

        assert (t_mels.size(1) - outs.size(1)) < 3, f"{outs.size(1)}, {t_mels.size(1)}"
        assert (t_mels.size(1) - outs.size(1)) >= 0, f"{outs.size(1)}, {t_mels.size(1)}"
        t_mels = t_mels[:, :outs.size(1), :]

        stop_flags = torch.zeros(t_mels.size(0), t_mels.size(1)).to(t_mels.device)
        for idx, out_len in enumerate(t_mel_lens):
            stop_flags[idx, int(out_len) - 1:] = 1.0

        t_mels = t_mels.masked_select(t_mel_masks.unsqueeze(-1))
        outs = outs.masked_select(t_mel_masks.unsqueeze(-1))
        outs_fine = outs_fine.masked_select(t_mel_masks.unsqueeze(-1))
        stop_flags = stop_flags.masked_select(t_mel_masks)
        logits = logits.masked_select(t_mel_masks)

        # Loss
        decoder_out_loss = self.mse_loss(outs, t_mels)
        postnet_out_loss = self.mse_loss(outs_fine, t_mels)
        stop_token_loss = self.bce_loss(logits, stop_flags)
        total_loss = decoder_out_loss + postnet_out_loss + stop_token_loss

        loss_values = {
            "mel_loss": decoder_out_loss.item(),
            "postnet_mel_loss": postnet_out_loss.item(),
            "stoptoken_loss": stop_token_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_values
