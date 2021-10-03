from typing import Dict, Optional

import torch.nn as nn
import torch

from vc_tts_template.fastspeech2VC.encoder_decoder import Encoder, Decoder
from vc_tts_template.fastspeech2VC.layers import PostNet
from vc_tts_template.fastspeech2VC.varianceadaptor import VarianceAdaptor, LengthRegulator
from vc_tts_template.fastspeech2wGMM.prosody_model import ProsodyExtractor
from vc_tts_template.fastspeech2VCwGMM.prosody_model import ProsodyPredictor
from vc_tts_template.utils import make_pad_mask
from vc_tts_template.train_utils import free_tensors_memory


class fastspeech2VCwGMM(nn.Module):
    """ FastSpeech2 """

    def __init__(
        self,
        n_mel_channel,
        # enoder_decoder
        attention_dim: int,
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        decoder_hidden_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        conv_kernel_size: int,
        ff_expansion_factor: int,
        conv_expansion_factor: int,
        ff_dropout: float,
        attention_dropout: float,
        conv_dropout: float,
        # prosody extractor
        prosody_emb_dim: int,
        extra_conv_kernel_size: int,
        extra_conv_n_layers: int,
        extra_gru_n_layers: int,
        extra_global_gru_n_layers: int,
        # prosody predictor
        gru_hidden_dim: int,
        gru_n_layers: int,
        pp_conv_out_channels: int,
        pp_conv_kernel_size: int,
        pp_conv_n_layers: int,
        pp_conv_dropout: float,
        pp_zoneout: float,
        num_gaussians: int,
        global_gru_n_layers: int,
        global_d_gru: int,
        global_num_gaussians: int,
        # varianceadaptor
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size_d: int,
        variance_predictor_layer_num_d: int,
        variance_predictor_kernel_size_p: int,
        variance_predictor_layer_num_p: int,
        variance_predictor_kernel_size_e: int,
        variance_predictor_layer_num_e: int,
        variance_predictor_dropout: int,
        stop_gradient_flow_d: bool,
        stop_gradient_flow_p: bool,
        stop_gradient_flow_e: bool,
        reduction_factor: int,

        encoder_fix: bool,
        global_prosody: bool,
        pitch_AR: bool = False,
        lstm_layers: int = 2,
        speakers: Optional[Dict] = None,
        emotions: Optional[Dict] = None
    ):
        super(fastspeech2VCwGMM, self).__init__()
        self.reduction_factor = reduction_factor
        self.mel_num = n_mel_channel

        self.encoder = Encoder(
            encoder_hidden_dim, attention_dim, encoder_num_layer, encoder_num_head,
            conv_kernel_size, ff_expansion_factor, conv_expansion_factor,
            ff_dropout, attention_dropout, conv_dropout
        )
        self.prosody_extractor = ProsodyExtractor(
            n_mel_channel,
            prosody_emb_dim,
            conv_kernel_size=extra_conv_kernel_size,
            conv_n_layers=extra_conv_n_layers,
            gru_n_layers=extra_gru_n_layers,
            global_prosody=global_prosody,
            global_gru_n_layers=extra_global_gru_n_layers,
        )
        self.prosody_predictor = ProsodyPredictor(
            encoder_hidden_dim,
            gru_hidden_dim,
            prosody_emb_dim,
            pp_conv_out_channels,
            conv_kernel_size=pp_conv_kernel_size,
            conv_n_layers=pp_conv_n_layers,
            conv_dropout=pp_conv_dropout,
            gru_layers=gru_n_layers,
            zoneout=pp_zoneout,
            num_gaussians=num_gaussians,
            global_prosody=global_prosody,
            global_gru_layers=global_gru_n_layers,
            global_d_gru=global_d_gru,
            global_num_gaussians=global_num_gaussians,
        )
        self.variance_adaptor = VarianceAdaptor(
            encoder_hidden_dim, variance_predictor_filter_size, variance_predictor_kernel_size_d,
            variance_predictor_layer_num_d, variance_predictor_kernel_size_p, variance_predictor_layer_num_p,
            variance_predictor_kernel_size_e, variance_predictor_layer_num_e, variance_predictor_dropout,
            stop_gradient_flow_d, stop_gradient_flow_p, stop_gradient_flow_e, reduction_factor, pitch_AR, lstm_layers
        )
        self.decoder = Decoder(
            decoder_hidden_dim, attention_dim, decoder_num_layer, decoder_num_head,
            conv_kernel_size, ff_expansion_factor, conv_expansion_factor,
            ff_dropout, attention_dropout, conv_dropout
        )
        self.mel_linear_1 = nn.Linear(
            self.mel_num * self.reduction_factor,
            encoder_hidden_dim,
        )
        self.prosody_linear = nn.Linear(
            prosody_emb_dim,
            encoder_hidden_dim,
        )
        self.prosody_length_regulator = LengthRegulator()
        self.decoder_linear = nn.Linear(
            encoder_hidden_dim,
            decoder_hidden_dim,
        )
        self.mel_linear_2 = nn.Linear(
            decoder_hidden_dim,
            self.mel_num * self.reduction_factor,
        )
        self.postnet = PostNet(
            n_mel_channels=self.mel_num
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
        self.global_prosody = global_prosody
        self.speakers = speakers
        self.emotions = emotions

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
        s_pitches,
        s_energies,
        t_mels=None,
        t_mel_lens=None,
        max_t_mel_len=None,
        t_pitches=None,
        t_energies=None,
        t_durations=None,
        s_snt_durations=None,
        t_snt_durations=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        s_mel_masks = make_pad_mask(s_mel_lens, max_s_mel_len)
        t_mel_masks = (
            make_pad_mask(t_mel_lens, max_t_mel_len)
            if t_mel_lens is not None
            else None
        )
        if s_snt_durations is None:
            s_snt_durations = torch.Tensor([s_mel.size(0) for s_mel in s_mels]).long()

        if self.reduction_factor > 1:
            max_s_mel_len = max_s_mel_len // self.reduction_factor
            s_mel_lens = torch.trunc(s_mel_lens / self.reduction_factor)
            s_mel_masks = make_pad_mask(s_mel_lens, max_s_mel_len)
            s_mels = s_mels[:, :max_s_mel_len*self.reduction_factor, :]

            if t_mel_lens is not None:
                max_t_mel_len = max_t_mel_len // self.reduction_factor
                t_mel_lens = torch.trunc(t_mel_lens / self.reduction_factor)
                t_mel_masks = make_pad_mask(t_mel_lens, max_t_mel_len)
                t_mels = t_mels[:, ::self.reduction_factor, :]

        output = self.mel_linear_1(
            s_mels.contiguous().view(s_mels.size(0), -1, self.mel_num * self.reduction_factor)
        )
        free_tensors_memory([s_mels])

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(s_sp_ids).unsqueeze(1).expand(-1, output.size(1), -1)
        if self.emotion_emb is not None:
            output = output + self.emotion_emb(s_em_ids).unsqueeze(1).expand(-1, output.size(1), -1)

        output = self.encoder(output, s_mel_masks)

        if self.encoder_fix is True:
            output = output.detach()

        is_inference = True if t_pitches is None else False

        if self.global_prosody is False:
            prosody_target = self.prosody_extractor(t_mels, t_snt_durations)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, snt_mask = self.prosody_predictor(
                output, s_snt_durations, target_prosody=prosody_target, is_inference=is_inference
            )
        else:
            prosody_target, g_prosody_target = self.prosody_extractor(t_mels, t_snt_durations)
            prosody_prediction, pi_outs, sigma_outs, mu_outs, snt_mask, g_pi, g_sigma, g_mu = self.prosody_predictor(
                output, s_snt_durations, target_prosody=prosody_target, target_global_prosody=g_prosody_target,
                is_inference=is_inference
            )
        if is_inference is True:
            prosody_prediction_expanded, _ = self.prosody_length_regulator(prosody_prediction, s_snt_durations)
            output = output + self.prosody_linear(prosody_prediction_expanded)
            free_tensors_memory([prosody_prediction, prosody_prediction_expanded, s_snt_durations])
        else:
            prosody_target_expanded, _ = self.prosody_length_regulator(prosody_target, s_snt_durations)
            output = output + self.prosody_linear(prosody_target_expanded)
            free_tensors_memory([prosody_prediction, prosody_target_expanded, s_snt_durations])
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            t_mel_lens,
            t_mel_masks,
        ) = self.variance_adaptor(
            output,
            s_mel_masks,
            max_s_mel_len,
            s_pitches,
            s_energies,
            t_durations,
            t_mel_masks,
            max_t_mel_len,
            t_pitches,
            t_energies,
            p_control,
            e_control,
            d_control,
        )

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(t_sp_ids).unsqueeze(1).expand(-1, output.size(1), -1)
        if self.emotion_emb is not None:
            output = output + self.speaker_emb(t_em_ids).unsqueeze(1).expand(-1, output.size(1), -1)

        output = self.decoder(self.decoder_linear(output), t_mel_masks)
        output = self.mel_linear_2(output).contiguous().view(output.size(0), -1, self.mel_num)

        postnet_output = self.postnet(output) + output

        t_mel_lens *= self.reduction_factor
        t_mel_masks = make_pad_mask(t_mel_lens, torch.max(t_mel_lens).item())

        if self.global_prosody is True:
            return (
                output,
                postnet_output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                s_mel_masks,
                t_mel_masks,
                s_mel_lens,
                t_mel_lens,
                prosody_target,
                pi_outs,
                mu_outs,
                sigma_outs,
                snt_mask,
                g_prosody_target,
                g_pi,
                g_mu,
                g_sigma,
            )
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            s_mel_masks,
            t_mel_masks,
            s_mel_lens,
            t_mel_lens,
            prosody_target,
            pi_outs,
            mu_outs,
            sigma_outs,
            snt_mask,
        )
