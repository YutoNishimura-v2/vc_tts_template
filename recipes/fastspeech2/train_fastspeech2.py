import sys
from functools import partial

import hydra
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.append("../..")
from vc_tts_template.fastspeech2.collate_fn import (
    collate_fn_fastspeech2, fastspeech2_get_data_loaders)
from vc_tts_template.frontend.openjtalk import sequence_to_text
from vc_tts_template.train_utils import plot_2d_feats, plot_attention, setup
from recipes.common.train_loop import train_loop


@torch.no_grad()
def eval_model(
    step, model, writer, batch, is_inference
):
    # 最大3つまで
    N = min(len(batch[0]), 3)

    if is_inference:
        outs, outs_fine, att_ws, out_lens = [], [], [], []
        for idx in range(N):  # 1個ずつ処理するのがinference.
            out, out_fine, _, att_w = model.inference(in_feats[idx][: in_lens[idx]])
            outs.append(out)
            outs_fine.append(out_fine)
            att_ws.append(att_w)
            out_lens.append(len(out))
    else:
        outs, outs_fine, _, att_ws = model(in_feats, in_lens, out_feats)

    for idx in range(N):  # 一個ずつsummary writerしていく.
        text = "".join(
            sequence_to_text(in_feats[idx][: in_lens[idx]].cpu().data.numpy())
        )
        if is_inference:
            group = f"utt{idx+1}_inference"
        else:
            group = f"utt{idx+1}_teacher_forcing"

        out = outs[idx][: out_lens[idx]]
        out_fine = outs_fine[idx][: out_lens[idx]]
        rf = model.decoder.reduction_factor  # selfに保存しているので.
        att_w = att_ws[idx][: out_lens[idx] // rf, : in_lens[idx]]  # reduction factor分しかattnはないのであった.
        fig = plot_attention(att_w)
        writer.add_figure(f"{group}/attention", fig, step)
        plt.close()
        fig = plot_2d_feats(out, text)
        writer.add_figure(f"{group}/out_before_postnet", fig, step)
        plt.close()
        fig = plot_2d_feats(out_fine, text)
        writer.add_figure(f"{group}/out_after_postnet", fig, step)
        plt.close()
        if not is_inference:
            out_gt = out_feats[idx][: out_lens[idx]]
            fig = plot_2d_feats(out_gt, text)
            writer.add_figure(f"{group}/out_ground_truth", fig, step)
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
        collate_fn_fastspeech2, speaker_dict=config.model.netG.speakers
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writer, logger = setup(
        config, device, collate_fn, fastspeech2_get_data_loaders  # type: ignore
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss, data_loaders, writer, logger, eval_model)


if __name__ == "__main__":
    my_app()
