import sys
from functools import partial
import warnings

import hydra
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
sys.path.append("../..")
from vc_tts_template.tacotronVC.collate_fn import collate_fn_tacotron2VC
from vc_tts_template.train_utils import (
    setup, get_vocoder, vocoder_infer, free_tensors_memory, plot_mels, plot_attention
)
from recipes.common.train_loop import train_loop

warnings.simplefilter('ignore', UserWarning)


def tacotron2VC_train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    loss,
    batch,
    logger,
    scaler,
    grad_checker,
):
    """dev時にはpredしたp, eで計算してほしいので, オリジナルのtrain_stepに.
    """
    optimizer.zero_grad()

    # Run forwaard
    with torch.cuda.amp.autocast():
        if train is True:
            output = model(*batch)
        else:
            # tacotron2VC
            output = model.inference(*batch)

        loss, loss_values = loss(batch, output)
    # Update
    if train:
        scaler.scale(loss).backward()
        grad_checker.set_params(model.named_parameters())
        free_tensors_memory([loss])
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            grad_checker.report(loss_values)
            if scaler.is_enabled() is True:
                logger.info("grad norm is NaN. Will Skip updating")
            else:
                logger.error("grad norm is NaN. check your model grad flow.")
                raise ValueError("Please check log.")
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

    return loss_values


@torch.no_grad()
def tacotron2VC_eval_model(
    phase, step, model, writer, batch, is_inference, vocoder_infer, sampling_rate, reduction_factor
):
    # 最大3つまで
    N = min(len(batch[0]), 3)

    if is_inference:
        # tacotron2VC
        output = model.inference(
            *batch[:8],
            t_mels=None,
            t_mel_lens=None,
            max_t_mel_len=None,
            max_synth_num=N,
        )
    else:
        output = model(*batch)

    for idx in range(N):
        file_name = batch[0][idx]
        if phase == 'train':
            file_name = f"utt_{idx}"

        mel_len_pre = int(output[4][idx].item())
        mel_post = output[1][idx][:mel_len_pre].cpu().data.numpy().T

        src_mel_len = int(batch[6][idx].item())
        mel_len_gt = int(batch[9][idx].item())
        mel_gt = batch[8][idx][:mel_len_gt].cpu().data.numpy().T
        audio_recon = vocoder_infer(batch[8][idx][:mel_len_gt].unsqueeze(0))[0]
        audio_synth = vocoder_infer(output[1][idx][:mel_len_pre].unsqueeze(0))[0]

        att_w = output[3][idx][:mel_len_pre // reduction_factor, :src_mel_len // reduction_factor]

        if is_inference:
            group = f"{phase}/inference"
        else:
            group = f"{phase}/teacher_forcing"

        fig = plot_attention(att_w)
        writer.add_figure(f"{group}/{file_name}_attention", fig, step)
        for i, argmax_idx in enumerate(torch.argmax(att_w, dim=-1)):
            att_w[i, :] = 0
            att_w[i, argmax_idx] = 1
        fig = plot_attention(att_w)
        writer.add_figure(f"{group}/{file_name}_attention_argmax", fig, step)
        fig = plot_mels([mel_post, mel_gt], ["out_after_postnet", "out_ground_truth"])
        writer.add_figure(f"{group}/{file_name}", fig, step)
        if is_inference:
            writer.add_audio(f"{group}/{file_name}_reconstruct", audio_recon/max(abs(audio_recon)), step, sampling_rate)
        writer.add_audio(f"{group}/{file_name}_synthesis", audio_synth/max(abs(audio_synth)), step, sampling_rate)
        plt.close()


def to_device(data, phase, device):
    (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        max_src_len,
        t_mels,
        t_mel_lens,
        max_mel_len,
    ) = data

    s_speakers = torch.from_numpy(s_speakers).long().to(device)
    t_speakers = torch.from_numpy(t_speakers).long().to(device)
    s_emotions = torch.from_numpy(s_emotions).long().to(device)
    t_emotions = torch.from_numpy(t_emotions).long().to(device)
    s_mels = torch.from_numpy(s_mels).float().to(device)
    s_mel_lens = torch.from_numpy(s_mel_lens).long().to(device)
    t_mels = torch.from_numpy(t_mels).float().to(device)
    t_mel_lens = torch.from_numpy(t_mel_lens).long().to(device)

    return (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        max_src_len,
        t_mels,
        t_mel_lens,
        max_mel_len,
    )


@hydra.main(config_path="conf/train_tacotron2VC", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 以下自由
    collate_fn = partial(
        collate_fn_tacotron2VC, batch_size=config.data.batch_size,
        speaker_dict=config.model.netG.speakers, emotion_dict=config.model.netG.emotions,
    )

    model, optimizer, lr_scheduler, loss, data_loaders, writers, logger, last_epoch, last_train_iter = setup(
        config, device, collate_fn  # type: ignore
    )
    # set_vocoder
    vocoder = get_vocoder(
        device, config.train.vocoder_name, config.train.vocoder_config, config.train.vocoder_weight_path
    )
    _vocoder_infer = partial(
        vocoder_infer, vocoder_dict={config.train.vocoder_name: vocoder},
        mel_scaler_path=config.train.mel_scaler_path,
        max_wav_value=config.train.max_wav_value
    )
    eval_model = partial(
        tacotron2VC_eval_model, vocoder_infer=_vocoder_infer, sampling_rate=config.train.sampling_rate,
        reduction_factor=config.model.netG.reduction_factor
    )

    # 以下固定
    to_device_ = partial(to_device, device=device)
    train_loop(config, to_device_, model, optimizer, lr_scheduler, loss,
               data_loaders, writers, logger, eval_model, tacotron2VC_train_step,
               last_epoch=last_epoch, last_train_iter=last_train_iter)


if __name__ == "__main__":
    my_app()
