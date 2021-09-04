from typing import List, Optional
import torch
import random
import numpy as np


def make_pad_mask(lengths: List, maxlen: Optional[int] = None) -> torch.Tensor:
    """Make mask for padding frames
    つまり, padした部分がTrueになっているようなmaskを返す.

    Args:
        lengths: list of lengths
        maxlen: maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def make_non_pad_mask(lengths: List, maxlen: Optional[int] = None) -> torch.Tensor:
    """Make mask for non-padding frames

    Args:
        lengths: list of lengths
        maxlen: maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask
    """
    return ~make_pad_mask(lengths, maxlen)


def load_utt_list(utt_list: str) -> List[str]:
    """Load a list of utterances.

    Args:
        utt_list: path to a file containing a list of utterances
        あくまでファイル名を返す.

    Returns:
        List[str]: list of utterances
    """
    utt_ids = []
    with open(utt_list) as f:
        for utt_id in f:
            utt_id = utt_id.strip()
            if len(utt_id) > 0:
                utt_ids.append(utt_id)
    return utt_ids


def init_seed(seed: int) -> None:
    """Initialize random seed.

    Args:
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_1d(x: torch.Tensor, max_len: int, constant_values: Optional[int] = 0) -> torch.Tensor:
    """Pad a 1d-tensor.

    Args:
        x: tensor to pad
        max_len: maximum length of the tensor
        constant_values: value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        (0, max_len - len(x)),
        mode="constant",
        constant_values=constant_values,
    )
    return x


def pad_2d(x: torch.Tensor, max_len: int, constant_values: Optional[int] = 0) -> torch.Tensor:
    """Pad a 2d-tensor.

    Args:
        x: tensor to pad
        max_len: maximum length of the tensor
        constant_values: value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x
