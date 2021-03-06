from functools import partial
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from vc_tts_template.train_utils import (
    get_epochs_with_optional_tqdm,
    moving_average_,
    save_checkpoint_old,
    setup_old,
)
from vc_tts_template.wavenet.collate_fn import collate_fn_wavenet


def train_step(model, model_ema, optimizer, lr_scheduler, train, criterion, x, c):
    optimizer.zero_grad()

    # 順伝播
    y_hat = model(x, c)

    # 損失 (負の対数尤度) の計算
    # y_hat: (B, C, T)
    # x: (B, T)
    # あくまで次の時刻からの予測なので, そこをずらす.
    loss = criterion(y_hat[:, :, :-1], x[:, 1:]).mean()

    # 逆伝播 & モデルパラメータの更新
    if train:
        loss.backward()
        optimizer.step()
        moving_average_(model, model_ema, beta=0.9999)
        lr_scheduler.step()

    return loss


def train_loop(
    config,
    logger,
    device,
    model,
    model_ema,
    optimizer,
    lr_scheduler,
    data_loaders,
    writer,
):
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max
    train_iter = 1
    nepochs = config.train.nepochs

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            for x, c in data_loaders[phase]:
                x, c = x.to(device), c.to(device)
                loss = train_step(
                    model,
                    model_ema,
                    optimizer,
                    lr_scheduler,
                    train,
                    criterion,
                    x,
                    c,
                )
                if train:
                    writer.add_scalar("LossByStep/train", loss.item(), train_iter)
                    train_iter += 1
                running_loss += loss.item()

            ave_loss = running_loss / len(data_loaders[phase])
            writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint_old(logger, out_dir, model, optimizer, epoch, True)
                save_checkpoint_old(
                    logger,
                    out_dir,
                    model_ema,
                    optimizer,
                    epoch,
                    True,
                    "_ema",
                )

        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint_old(logger, out_dir, model, optimizer, epoch, False)
            save_checkpoint_old(
                logger,
                out_dir,
                model_ema,
                optimizer,
                epoch,
                False,
                "_ema",
            )

    # save at last epoch
    save_checkpoint_old(logger, out_dir, model, optimizer, nepochs)
    save_checkpoint_old(logger, out_dir, model_ema, optimizer, nepochs, False, "_ema")


@hydra.main(config_path="conf/train_wavenet", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate_fn = partial(
        collate_fn_wavenet,
        max_time_frames=config.data.max_time_frames,
        hop_size=config.data.hop_size,
        aux_context_window=config.data.aux_context_window,
    )
    model, optimizer, lr_scheduler, data_loaders, writer, logger = setup_old(
        config, device, collate_fn  # type: ignore
    )
    # exponential moving average
    model_ema = hydra.utils.instantiate(config.model.netG).to(device)
    state_dict = (
        model.module.state_dict() if config.data_parallel else model.state_dict()
    )
    model_ema.load_state_dict(state_dict)

    # Run training loop
    train_loop(
        config,
        logger,
        device,
        model,
        model_ema,
        optimizer,
        lr_scheduler,
        data_loaders,
        writer,
    )


if __name__ == "__main__":
    my_app()
