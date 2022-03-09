from typing import Dict, Optional

import torch
import torch.nn as nn

from vc_tts_template.fastspeech2wContexts.fastspeech2wContextswPEProsody import FastSpeech2wContextswPEProsody


class FastSpeech2wContextswPEProsodywCurrentMel(FastSpeech2wContextswPEProsody):
    """ FastSpeech2wContexts """

    def __init__(
        self,
        max_seq_len: int,
        num_vocab: int,  # pad=0
        # encoder
        encoder_hidden_dim: int,
        encoder_num_layer: int,
        encoder_num_head: int,
        conv_filter_size: int,
        conv_kernel_size_1: int,
        conv_kernel_size_2: int,
        encoder_dropout: float,
        # context encoder
        context_encoder_hidden_dim: int,
        context_num_layer: int,
        context_encoder_dropout: float,
        text_emb_dim: int,
        peprosody_encoder_gru_dim: int,
        peprosody_encoder_gru_num_layer: int,
        shere_embedding: bool,
        current_attention: bool,
        past_global_gru: bool,
        mel_embedding_mode: int,
        pau_split_mode: int,
        # mel_emb_dim: int,
        # mel_emb_kernel: int,
        # mel_emb_dropout: float,
        peprosody_encoder_conv_kernel_size: int,
        peprosody_encoder_conv_n_layers: int,
        sslprosody_emb_dim: Optional[int],
        sslprosody_layer_num: Optional[int],
        use_context_encoder: bool,
        use_prosody_encoder: bool,
        use_peprosody_encoder: bool,
        use_melprosody_encoder: bool,
        last_concat: bool,
        # variance predictor
        variance_predictor_filter_size: int,
        variance_predictor_kernel_size: int,
        variance_predictor_dropout: int,
        pitch_feature_level: int,  # 0 is frame 1 is phoneme
        energy_feature_level: int,  # 0 is frame 1 is phoneme
        pitch_quantization: str,
        energy_quantization: str,
        pitch_embed_kernel_size: int,
        pitch_embed_dropout: float,
        energy_embed_kernel_size: int,
        energy_embed_dropout: float,
        n_bins: int,
        # decoder
        decoder_hidden_dim: int,
        decoder_num_layer: int,
        decoder_num_head: int,
        decoder_dropout: float,
        n_mel_channel: int,
        # other
        encoder_fix: bool,
        stats: Optional[Dict],
        speakers: Dict,
        emotions: Optional[Dict] = None,
        accent_info: int = 0,
    ):
        super().__init__(
            max_seq_len,
            num_vocab,
            encoder_hidden_dim,
            encoder_num_layer,
            encoder_num_head,
            conv_filter_size,
            conv_kernel_size_1,
            conv_kernel_size_2,
            encoder_dropout,
            context_encoder_hidden_dim,
            context_num_layer,
            context_encoder_dropout,
            text_emb_dim,
            peprosody_encoder_gru_dim,
            peprosody_encoder_gru_num_layer,
            shere_embedding,
            current_attention,
            past_global_gru,
            mel_embedding_mode,
            pau_split_mode,
            peprosody_encoder_conv_kernel_size,
            peprosody_encoder_conv_n_layers,
            sslprosody_emb_dim,
            sslprosody_layer_num,
            use_context_encoder,
            use_prosody_encoder,
            use_peprosody_encoder,
            use_melprosody_encoder,
            last_concat,
            variance_predictor_filter_size,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            pitch_feature_level,
            energy_feature_level,
            pitch_quantization,
            energy_quantization,
            pitch_embed_kernel_size,
            pitch_embed_dropout,
            energy_embed_kernel_size,
            energy_embed_dropout,
            n_bins,
            decoder_hidden_dim,
            decoder_num_layer,
            decoder_num_head,
            decoder_dropout,
            n_mel_channel,
            encoder_fix,
            stats,
            speakers,
            emotions,
            accent_info,
        )
        if sslprosody_emb_dim is not None:
            self.next_predictor = nn.Sequential(
                nn.Linear(encoder_hidden_dim, sslprosody_emb_dim),
                nn.ReLU(),
                nn.Linear(sslprosody_emb_dim, sslprosody_emb_dim),
                nn.ReLU(),
                nn.Linear(sslprosody_emb_dim, sslprosody_emb_dim),
                nn.ReLU(),
            )
        else:
            raise RuntimeError("未対応です")

        self.current_attention = current_attention
        if (use_context_encoder is True) and (use_melprosody_encoder is True):
            self.club_estimator = CLUBSample(
                encoder_hidden_dim, encoder_hidden_dim, encoder_hidden_dim * 2
            )
        else:
            self.club_estimator = None  # type: ignore

    def contexts_forward(
        self,
        output,
        max_src_len,
        c_txt_embs,
        c_txt_embs_lens,
        speakers,
        emotions,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        h_prosody_embs,
        h_prosody_embs_lens,
        h_prosody_embs_len,
        c_prosody_embs_phonemes,
    ):
        if (self.use_peprosody_encoder or self.use_melprosody_encoder) is True:
            if self.use_ssl is False:
                h_prosody_emb = self.peprosody_encoder(
                    h_prosody_embs,
                    h_prosody_embs_lens,
                )
            else:
                # h_prosody_embs: (B, hist_len, layer_num, dim)
                if self.peprosody_encoder is not None:
                    batch_size = h_prosody_embs.size(0)
                    history_len = h_prosody_embs.size(1)
                    if h_prosody_embs.size(-2) == 1:
                        # batch全てPADのデータはこれになる
                        h_prosody_emb = h_prosody_embs.view(batch_size, history_len, -1)
                    else:
                        h_prosody_embs = h_prosody_embs.view(-1, h_prosody_embs.size(-2), h_prosody_embs.size(-1))
                        h_prosody_emb = self.peprosody_encoder(
                            h_prosody_embs
                        ).view(batch_size, history_len, -1)
                else:
                    h_prosody_emb = h_prosody_embs.squeeze(-2)

        else:
            h_prosody_emb = None

        context_enc_outputs = self.context_encoder(
            c_txt_embs,
            c_txt_embs_lens,
            speakers,
            emotions,
            h_txt_embs,
            h_txt_emb_lens,  # [hist1, hist2, ...]
            h_speakers,
            h_emotions,
            h_prosody_emb,
            h_prosody_embs_len,  # [hist1, hist2, ...]. h_txt_emb_lensとは違って1 start.
        )
        if type(context_enc_outputs) == tuple:
            if self.current_attention is True:
                if len(context_enc_outputs) == 3:
                    context_enc = context_enc_outputs[0]
                    attentions = context_enc_outputs[1:]
                    context_embs = None
                if len(context_enc_outputs) == 5:
                    context_enc = context_enc_outputs[0]
                    attentions = context_enc_outputs[1:3]
                    context_embs = context_enc_outputs[3:]
            else:
                context_enc = context_enc_outputs[0]
                attentions = None
                context_embs = context_enc_outputs[1:]
        else:
            context_enc = context_enc_outputs
            attentions = None
            context_embs = None

        if c_prosody_embs_phonemes is None:
            output = output + context_enc.unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        else:
            context_enc, _ = self.length_regulator(
                context_enc, c_prosody_embs_phonemes, torch.max(c_prosody_embs_phonemes)
            )
            output = output + context_enc

        # 以下だけ変更
        prosody_prediction = self.next_predictor(context_enc)

        return output, prosody_prediction, attentions, context_embs

    def forward(
        self,
        ids,
        speakers,
        emotions,
        texts,
        src_lens,
        max_src_len,
        c_txt_embs,
        c_txt_embs_lens,
        h_txt_embs,
        h_txt_emb_lens,
        h_speakers,
        h_emotions,
        c_prosody_embs,
        c_prosody_embs_lens,
        c_prosody_embs_duration,
        c_prosody_embs_phonemes,
        h_prosody_embs,
        h_prosody_embs_lens,
        h_prosody_embs_len,
        h_local_prosody_emb=None,
        h_local_prosody_emb_lens=None,
        h_local_prosody_speakers=None,
        h_local_prosody_emotions=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        q_theta_training=False,
    ):
        src_lens, max_src_len, src_masks, mel_lens, max_mel_len, mel_masks = self.init_forward(
            src_lens, max_src_len, mel_lens, max_mel_len
        )
        output = self.encoder_forward(
            texts, src_masks, max_src_len, speakers, emotions
        )
        output, prosody_prediction, attentions, context_embs = self.contexts_forward(
            output, max_src_len, c_txt_embs, c_txt_embs_lens,
            speakers, emotions,
            h_txt_embs, h_txt_emb_lens, h_speakers, h_emotions,
            h_prosody_embs, h_prosody_embs_lens, h_prosody_embs_len,
            c_prosody_embs_phonemes,
        )
        if (q_theta_training is True) and (self.club_estimator is not None):
            t_emb, p_emb = context_embs
            loss = self.club_estimator.learning_loss(
                t_emb, p_emb
            )
            return loss
        elif (q_theta_training is False) and (self.club_estimator is not None):
            t_emb, p_emb = context_embs
            mi = self.club_estimator(
                t_emb, p_emb
            )
        else:
            mi = None
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        output, postnet_output, mel_masks = self.decoder_forward(
            output, mel_masks
        )

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            prosody_prediction,
            attentions,
            mi,
        )


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def forward(self, x_samples, y_samples):
        # vCLUBの計算
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        # q_θ の訓練用のloss出力
        return - self.loglikeli(x_samples, y_samples)


class FastSpeech2wContextswPEProsodywCurrentMelsLoss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, pitch_feature_level, energy_feature_level, beta, g_beta=None):
        super().__init__()
        self.pitch_feature_level = "phoneme_level" if pitch_feature_level > 0 else "frame_level"
        self.energy_feature_level = "phoneme_level" if energy_feature_level > 0 else "frame_level"

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.beta = beta
        self.g_beta = g_beta

    def forward(self, inputs, predictions):
        (
            target_prosody_embs,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[-17:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            predicted_prosody_embs,
            _,
            mi,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        prosody_emb_loss = self.mse_loss(predicted_prosody_embs, target_prosody_embs)

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + self.beta*prosody_emb_loss

        # mutual_infomationを計算する
        if (self.g_beta is not None) and (mi is not None):
            total_loss = total_loss + self.g_beta * mi
            loss_values = {
                "club_loss": mi.item(),
                "prosody_emb_loss": prosody_emb_loss.item(),
                "mel_loss": mel_loss.item(),
                "postnet_mel_loss": postnet_mel_loss.item(),
                "pitch_loss": pitch_loss.item(),
                "energy_loss": energy_loss.item(),
                "duration_loss": duration_loss.item(),
                "total_loss": total_loss.item()
            }
        else:
            loss_values = {
                "prosody_emb_loss": prosody_emb_loss.item(),
                "mel_loss": mel_loss.item(),
                "postnet_mel_loss": postnet_mel_loss.item(),
                "pitch_loss": pitch_loss.item(),
                "energy_loss": energy_loss.item(),
                "duration_loss": duration_loss.item(),
                "total_loss": total_loss.item()
            }

        return total_loss, loss_values
