from recipes.fastspeech2.utils import plot_mel
from recipes.common.train_loop import train_loop
from vc_tts_template.train_utils import setup
from vc_tts_template.vocoder.hifigan.collate_fn import (
    collate_fn_hifigan, hifigan_get_data_loaders)
import sys
from functools import partial

import hydra
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

sys.path.append("../..")


def fastspeech2_train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    loss,
    batch,
    logger
):
    """dev時にはpredしたp, eで計算してほしいので, オリジナルのtrain_stepに.
    """
    optimizer.zero_grad()

    # Run forwaard
    if train is True:
        output = model(*batch)
    else:
        output = model(
            *batch[:8],
            p_targets=None,
            e_targets=None,
            d_targets=batch[10]
        )

    loss, loss_values = loss(batch, output)

    # Update
    if train:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            # こんなことあるんだ.
            logger.info("grad norm is NaN. Skip updating")
        else:
            optimizer.step()
        lr_scheduler.step()

    return loss_values


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
    if len(data) == 4:
        (
            ids,
            audios,
            mels,
            mel_losses
        ) = data

        audios = torch.Tensor(audios).float().to(device)
        mels = torch.Tensor(mels).float().to(device)
        mel_losses = torch.Tensor(mel_losses).float().to(device)

        return (
            ids,
            audios,
            mels,
            mel_losses,
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

    collate_fn = partial(
        collate_fn_hifigan, config=config.data
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger = setup(
        config, device, collate_fn, hifigan_get_data_loaders  # type: ignore
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss, data_loaders,
               writers, logger, eval_model, fastspeech2_train_step)


if __name__ == "__main__":
    my_app()
