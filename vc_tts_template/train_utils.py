import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Iterable
import shutil

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from vc_tts_template.logger import getLogger
from vc_tts_template.utils import init_seed, load_utt_list


def get_epochs_with_optional_tqdm(tqdm_mode: str, nepochs: int) -> Iterable:
    """Get epochs with optional progress bar.

    Args:
        tqdm_mode: Progress bar mode.
          tqdmと書くか, それ以外か.
        nepochs: Number of epochs.

    Returns:
        iterable: Epochs.
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm

        epochs = tqdm(range(1, nepochs + 1), desc="epoch")
    else:
        epochs = range(1, nepochs + 1)

    return epochs


def num_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in the model.

    Args:
        model: Model to count the number of trainable parameters.

    Returns:
        int: Number of trainable parameters.
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def set_epochs_based_on_max_steps_(train_config: Dict, steps_per_epoch: int, logger: logging.Logger):
    """Set epochs based on max steps.

    Args:
        train_config: Train config.
        steps_per_epoch: Number of steps per epoch.
        logger: Logger.
    """
    logger.info(f"Number of iterations per epoch: {steps_per_epoch}")

    if train_config.max_train_steps < 0:  # type: ignore
        # Set max_train_steps based on nepochs
        max_train_steps = train_config.nepochs * steps_per_epoch  # type: ignore
        train_config.max_train_steps = max_train_steps  # type: ignore
        logger.info(
            "Number of max_train_steps is set based on nepochs: {}".format(
                max_train_steps
            )
        )
    else:
        # Set nepochs based on max_train_steps
        max_train_steps = train_config.max_train_steps  # type: ignore
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        train_config.nepochs = epochs  # type: ignore
        logger.info(
            "Number of epochs is set based on max_train_steps: {}".format(epochs)
        )

    logger.info(f"Number of epochs: {train_config.nepochs}")  # type: ignore
    logger.info(f"Number of iterations: {train_config.max_train_steps}")  # type: ignore


def save_checkpoint(
    logger: logging.Logger, out_dir: Path, model: nn.Module, optimizer: optim.Optimizer,
    epoch: int, is_best: Optional[bool] = False, postfix: Optional[str] = ""
) -> None:
    """Save a checkpoint.

    Args:
        logger: Logger.
        out_dir: Output directory.
        model: Model.
        optimizer: Optimizer.
        epoch: Current epoch.
        is_best: Whether or not the current model is the best.
            Defaults to False.
        postfix: Postfix. Defaults to "".
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def ensure_divisible_by(feats: np.ndarray, N: int) -> np.ndarray:
    """Ensure that the number of frames is divisible by N.
    Args:
        feats (np.ndarray): Input features.
        N (int): Target number of frames.
    Returns:
        np.ndarray: Input features with number of frames divisible by N.
    """
    if N == 1:
        return feats
    mod = len(feats) % N
    if mod != 0:
        feats = feats[: len(feats) - mod]
    return feats


def moving_average_(model, model_test, beta=0.9999):
    """Exponential moving average (EMA) of model parameters.

    Args:
        model (torch.nn.Module): Model to perform EMA on.
        model_test (torch.nn.Module): Model to use for the test phase.
        beta (float, optional): [description]. Defaults to 0.9999.
    """
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def plot_attention(alignment: torch.Tensor) -> plt.figure:
    """Plot attention.
    Args:
        alignment (np.ndarray): Attention.
    """
    fig, ax = plt.subplots()
    alignment = alignment.cpu().data.numpy().T
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    plt.xlabel("Decoder time step")
    plt.ylabel("Encoder time step")
    return fig


def plot_2d_feats(feats: torch.Tensor, title=None) -> plt.figure:
    """Plot 2D features.
    Args:
        feats (np.ndarray): Input features.
        title (str, optional): Title. Defaults to None.
    """
    feats = feats.cpu().data.numpy().T
    fig, ax = plt.subplots()
    im = ax.imshow(
        feats, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)
    return fig


class _Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths: List of paths to input files
        out_paths: List of paths to output files
    """

    def __init__(self, in_paths: List, out_paths: List):
        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return np.load(self.in_paths[idx]), np.load(self.out_paths[idx])

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)


def _get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
    """Get data loaders for training and validation.

    Args:
        data_config: Data configuration.
        collate_fn: Collate function.

    Returns:
        dict: Data loaders.
    """
    data_loaders = {}

    for phase in ["train", "dev"]:
        utt_ids = load_utt_list(to_absolute_path(data_config[phase].utt_list))
        in_dir = Path(to_absolute_path(data_config[phase].in_dir))
        out_dir = Path(to_absolute_path(data_config[phase].out_dir))

        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_feats_paths = [out_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        dataset = _Dataset(in_feats_paths, out_feats_paths)
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size,  # type: ignore
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,  # type: ignore
            shuffle=phase.startswith("train"),  # trainならTrue
        )

    return data_loaders


def setup(config: Dict, device: torch.device, collate_fn: Callable) -> Tuple:
    """Setup for traiining

    Args:
        config: configuration for training
        device: device to use for training
        collate_fn: function to collate mini-batches

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.

    .. note::

        書籍に記載のコードは、この関数を一部簡略化しています。
    """
    # NOTE: hydra は内部で stream logger を追加するので、二重に追加しないことに注意
    logger = getLogger(config.verbose, add_stream_handler=False)  # type: ignore

    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA 周りの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark  # type: ignore
        cudnn.deterministic = config.cudnn.deterministic  # type: ignore
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")  # type: ignore
    init_seed(config.seed)  # type: ignore

    # モデルのインスタンス化
    # ↓これでインスタンス化できるhydraすごすぎる.
    model = hydra.utils.instantiate(config.model.netG).to(device)  # type: ignore
    logger.info(model)
    logger.info(
        "Number of trainable params: {:.3f} million".format(
            num_trainable_params(model) / 1000000.0
        )
    )

    # (optional) 学習済みモデルの読み込み
    # ファインチューニングしたい場合
    pretrained_checkpoint = config.train.pretrained.checkpoint  # type: ignore
    if pretrained_checkpoint is not None and len(pretrained_checkpoint) > 0:
        logger.info(
            "Fine-tuning! Loading a checkpoint: {}".format(pretrained_checkpoint)
        )
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # 複数 GPU 対応
    if config.data_parallel:  # type: ignore
        model = nn.DataParallel(model)

    # Optimizer
    # 例: config.train.optim.optimizer.name = "Adam"なら,
    # optim.Adamをしていることと等価.
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)  # type: ignore
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params  # type: ignore
    )

    # 学習率スケジューラ
    lr_scheduler_class = getattr(
        optim.lr_scheduler, config.train.optim.lr_scheduler.name  # type: ignore
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **config.train.optim.lr_scheduler.params  # type: ignore
    )

    # DataLoader
    data_loaders = _get_data_loaders(config.data, collate_fn)  # type: ignore

    set_epochs_based_on_max_steps_(config.train, len(data_loaders["train"]), logger)  # type: ignore

    # Tensorboard の設定
    writer = SummaryWriter(to_absolute_path(config.train.log_dir))  # type: ignore

    # config ファイルを保存しておく
    # Pathめっちゃ有能じゃん
    out_dir = Path(to_absolute_path(config.train.out_dir))  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)  # type: ignore
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    return model, optimizer, lr_scheduler, data_loaders, writer, logger
