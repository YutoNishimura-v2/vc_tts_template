import sys
from functools import partial

import hydra
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.append("../..")
from vc_tts_template.fastspeech2.collate_fn import (
    collate_fn_fastspeech2, fastspeech2_get_data_loaders)
from vc_tts_template.train_utils import setup
from recipes.common.train_loop import train_loop
from recipes.fastspeech2.utils import plot_mel


@torch.no_grad()
def eval_model(
    phase, step, model, writer, batch, is_inference
):
    # 最大3つまで
    N = min(len(batch[0]), 3)

    if is_inference:
        # pitch, energyとして正解データを与えない.
        output = model(
            *batch[:8],
            p_targets=None,
            e_targets=None,
            d_targets=batch[10]
        )
    else:
        output = model(*batch)

    for idx in range(N):  # 一個ずつsummary writerしていく.
        file_name = batch[0][idx]
        if phase == 'train':
            file_name = f"utt_{idx}"
        mel_post = output[1][idx].cpu().data.numpy().T
        pitch = output[2][idx].cpu().data.numpy()
        energy = output[3][idx].cpu().data.numpy()
        duration = batch[10][idx].cpu().data.numpy()
        mel_gt = batch[5][idx].cpu().data.numpy().T
        pitch_gt = batch[8][idx].cpu().data.numpy()
        energy_gt = batch[9][idx].cpu().data.numpy()

        mel_gts = [mel_gt, pitch_gt, energy_gt, duration]

        if is_inference:
            group = f"{phase}/inference"
            mel_posts = [mel_post, pitch, energy, duration]
        else:
            group = f"{phase}/teacher_forcing"
            mel_posts = [mel_post, pitch_gt, energy_gt, duration]

        fig = plot_mel([mel_posts, mel_gts], ["out_after_postnet", "out_ground_truth"])
        writer.add_figure(f"{group}/{file_name}", fig, step)
        plt.close()


def to_device(data, device):
    if len(data) == 11:
        (
            ids,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


@hydra.main(config_path="conf/train_fastspeech2", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    if config.model.netG.speakers is None:
        assert config.model.netG.multi_speaker == 0, f"""multi_speaker: {config.model.netG.multi_speaker},
        speakers_dict: {config.model.netG.speakers}"""
    else:
        assert config.model.netG.multi_speaker > 0, f"""multi_speaker: {config.model.netG.multi_speaker},
        speakers_dict: {config.model.netG.speakers}"""

    collate_fn = partial(
        collate_fn_fastspeech2, batch_size=config.data.batch_size, speaker_dict=config.model.netG.speakers
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger = setup(
        config, device, collate_fn, fastspeech2_get_data_loaders  # type: ignore
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, eval_model)


if __name__ == "__main__":
    my_app()
