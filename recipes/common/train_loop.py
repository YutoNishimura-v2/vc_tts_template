import torch
from hydra.utils import to_absolute_path
from tqdm import tqdm
from pathlib import Path
from vc_tts_template.train_utils import (
    get_epochs_with_optional_tqdm,
    save_checkpoint
)


def train_step(
    model,
    optimizer,
    lr_scheduler,
    train,
    loss,
    batch,
    logger
):
    optimizer.zero_grad()

    # Run forwaard
    output = model(*batch)

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


def _update_running_losses_(running_losses, loss_values):
    for key, val in loss_values.items():
        try:
            running_losses[key] += val
        except KeyError:
            running_losses[key] = val


def train_loop(config, to_device, model, optimizer, lr_scheduler, loss, data_loaders, writer, logger, eval_model):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_loss = torch.finfo(torch.float32).max
    train_iter = 1
    nepochs = config.train.nepochs

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_losses = {}  # epoch毎のloss. ここでresetしてるし.
            for idx, batch in tqdm(
                enumerate(data_loaders[phase]), desc=f"{phase} iter", leave=False
            ):
                batch = to_device(batch)

                loss_values = train_step(
                    model,
                    optimizer,
                    lr_scheduler,
                    train,
                    loss,
                    batch,
                    logger
                )
                if train:
                    for key, val in loss_values.items():
                        writer.add_scalar(f"{key}ByStep/train", val, train_iter)
                    writer.add_scalar(
                        "LearningRate", lr_scheduler.get_last_lr()[0], train_iter
                    )
                    train_iter += 1
                # lossを一気に足してためておく. 賢い.
                _update_running_losses_(running_losses, loss_values)

                # 最初の検証用データに対して、中間結果の可視化
                if (
                    not train
                    and idx == 0  # 最初
                    and epoch % config.train.eval_epoch_interval == 0
                ):
                    for is_inference in [False, True]:  # 非推論モードでやるの偉い.
                        eval_model(
                            train_iter,
                            model,
                            writer,
                            batch,
                            is_inference
                        )

            # Epoch ごとのロスを出力
            for key, val in running_losses.items():
                ave_loss = val / len(data_loaders[phase])
                writer.add_scalar(f"{key}/{phase}", ave_loss, epoch)

            ave_loss = running_losses["Loss"] / len(data_loaders[phase])
            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(logger, out_dir, model, optimizer, lr_scheduler, epoch, True)

        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(logger, out_dir, model, optimizer, lr_scheduler, epoch, False)

    # save at last epoch
    save_checkpoint(logger, out_dir, model, optimizer, lr_scheduler, nepochs)
    logger.info(f"The best loss was {best_loss}")

    return model
