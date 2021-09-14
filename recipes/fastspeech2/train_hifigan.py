import sys
from functools import partial

import hydra
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import torch.nn.functional as F

sys.path.append("../..")
from recipes.fastspeech2.utils import plot_mel
from recipes.common.train_loop import train_loop
from vc_tts_template.train_utils import setup
from vc_tts_template.vocoder.hifigan.collate_fn import (
    collate_fn_hifigan, hifigan_get_data_loaders)
from vc_tts_template.vocoder.hifigan.collate_fn import mel_spectrogram


def hifigan_train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    loss,
    batch,
    logger,
    mel_spectrogram_in_train_step
):
    if train:
        _, y, x, y_mel = batch
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)
        y_mel = torch.autograd.Variable(y_mel)
        y = y.unsqueeze(1)
        y_g_hat = model['netG'](x)
        y_g_hat_mel = mel_spectrogram_in_train_step(y=y_g_hat.squeeze(1))
        optimizer.optim_d.zero_grad()

        # MPD: multi period descriminator
        y_df_hat_r, y_df_hat_g, _, _ = model['netMPD'](y, y_g_hat.detach())
        loss_disc_f, _, _ = loss.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD: multi scale descriminator
        y_ds_hat_r, y_ds_hat_g, _, _ = model['netMSD'](y, y_g_hat.detach())
        loss_disc_s, _, _ = loss.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        optimizer.optim_d.step()

        optimizer.optim_g.zero_grad()

        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = model['netMPD'](y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = model['netMSD'](y, y_g_hat)
        loss_fm_f = loss.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = loss.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = loss.generator_loss(y_df_hat_g)
        loss_gen_s, _ = loss.generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        optimizer.optim_g.step()

        with torch.no_grad():
            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
        loss_values = {
            'Gen Loss Total': loss_gen_all.item(),
            'Mel-Spec. Error': mel_error
        }
        # 本来はepoch毎にstepしている(著者実装).
        lr_scheduler.scheduler_g.step()
        lr_scheduler.scheduler_d.step()

    else:
        torch.cuda.empty_cache()
        with torch.no_grad():
            # validationはbatch_size=1で固定.
            _, y, x, y_mel = batch
            y_g_hat = model['netG'](x)
            y_mel = torch.autograd.Variable(y_mel)
            y_g_hat_mel = mel_spectrogram(y=y_g_hat.squeeze(1))
            val_err = F.l1_loss(y_mel, y_g_hat_mel).item()
        loss_values = {
            'Mel-Spec. Error': val_err
        }
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
    (
        ids,
        audios,
        mels,
        mel_losses
    ) = data
    audios = torch.Tensor(audios).float().to(device, non_blocking=True)
    mels = torch.Tensor(mels).float().to(device, non_blocking=True)
    mel_losses = torch.Tensor(mel_losses).float().to(device, non_blocking=True)

    return (
        ids,
        audios,
        mels,
        mel_losses,
    )


@hydra.main(config_path="conf/train_hifigan", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    collate_fn = partial(
        collate_fn_hifigan, config=config.data
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger = setup(
        config, device, collate_fn, hifigan_get_data_loaders  # type: ignore
    )

    # train_stepで利用
    mel_spectrogram_in_train_step = partial(
        mel_spectrogram, n_fft=config.data.n_fft, num_mels=config.data.num_mels,
        sampling_rate=config.data.sampling_rate, hop_size=config.data.hop_size, 
        win_size=config.data.win_size, fmin=config.data.fmin, fmax=config.data.fmax_loss
    )
    train_step = partial(
        hifigan_train_step, mel_spectrogram_in_train_step=mel_spectrogram_in_train_step
    )
    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss, data_loaders,
               writers, logger, eval_model, train_step)


if __name__ == "__main__":
    my_app()
