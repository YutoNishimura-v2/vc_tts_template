from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pydub
import joblib
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from vc_tts_template.utils import adaptive_load_state_dict


def get_alignment_model(
    model_config_path: str, pretrained_checkpoint: str,
    device: torch.device, in_scaler_path: str, out_scaler_path: str
):
    checkpoint = torch.load(to_absolute_path(pretrained_checkpoint), map_location=device)
    model_config = OmegaConf.load(to_absolute_path(model_config_path))
    model = hydra.utils.instantiate(model_config.netG).to(device)
    adaptive_load_state_dict(model, checkpoint["state_dict"])
    acoustic_in_scaler = joblib.load(to_absolute_path(in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(out_scaler_path))
    model.eval()
    model.remove_weight_norm()

    return model, acoustic_in_scaler, acoustic_out_scaler


def get_alignment(
    model, acoustic_in_scaler, acoustic_out_scaler, src_mel, tgt_mel,
    s_sp_id=None, t_sp_id=None, s_em_id=None, t_em_id=None
):
    src_mel = acoustic_in_scaler.transform(src_mel)
    tgt_mel = acoustic_out_scaler.transform(tgt_mel)
    src_mel = torch.tensor(src_mel).unsqueeze(0).to(model.device)
    src_mel_lens = torch.tensor([src_mel.size(1)]).to(model.device)
    max_src_mel_len = src_mel.size(1)
    tgt_mel = torch.tensor(tgt_mel).unsqueeze(0).to(model.device)
    tgt_mel_lens = torch.tensor([tgt_mel.size(1)]).to(model.device)
    max_tgt_mel_len = tgt_mel.size(1)

    s_sp_id = model.speakers[s_sp_id] if s_sp_id is not None else 0
    t_sp_id = model.speakers[t_sp_id] if t_sp_id is not None else 0
    s_em_id = model.emotions[s_em_id] if s_em_id is not None else 0
    t_em_id = model.emotions[t_em_id] if t_em_id is not None else 0

    s_sp_ids = torch.tensor([s_sp_id], dtype=torch.long).to(model.device)
    t_sp_ids = torch.tensor([t_sp_id], dtype=torch.long).to(model.device)
    s_em_ids = torch.tensor([s_em_id], dtype=torch.long).to(model.device)
    t_em_ids = torch.tensor([t_em_id], dtype=torch.long).to(model.device)

    attn_w = model(
        None, s_sp_ids, t_sp_ids, s_em_ids, t_em_ids,
        src_mel, src_mel_lens, max_src_mel_len,
        tgt_mel, tgt_mel_lens, max_tgt_mel_len
    )[3][0].cpu().data.numpy()

    # 対角にあるとかは関係なく, encoder軸から見てargmaxになったものの個数をただ数える.
    # Fastspeech1の論文と同じ手法.
    duration = [np.sum(np.argmax(attn_w, axis=-1) == i) for i in range(attn_w.shape[1])]

    return np.array(duration)


def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """Converts pydub audio segment into float32 np array of shape [channels, duration_in_seconds*sample_rate],
    where each value is in range [-1.0, 1.0]. Returns tuple (audio_np_array, sample_rate)"""
    # get_array_of_samples returns the data in format:
    # [sample_1_channel_1, sample_1_channel_2, sample_2_channel_1, sample_2_channel_2, ....]
    # where samples are integers of sample_width bytes.
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def expand(values, durations, reduction_factor):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * (max(0, int(d))) * reduction_factor
    return np.array(out)


def plot_mel_with_prosody(data, titles, reduction_factor=1, need_expand=True):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy, duration = data[i]
        if need_expand is True:
            pitch = expand(pitch, duration, reduction_factor)
            energy = expand(energy, duration, reduction_factor)
        mel = mel[:, :np.sum(duration)*reduction_factor]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig
