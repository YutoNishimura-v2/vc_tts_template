import torch

from vc_tts_template.utils import pad_1d, pad_2d
from vc_tts_template.train_utils import ensure_divisible_by


def collate_fn_tacotron(batch, reduction_factor=1):
    """Collate function for Tacotron.
    Args:
        batch (list): List of tuples of the form (inputs, targets).
        reduction_factor (int, optional): Reduction factor. Defaults to 1.
    Returns:
        tuple: Batch of inputs, input lengths, targets, target lengths and stop flags.
    """
    xs = [x[0] for x in batch]
    ys = [ensure_divisible_by(x[1], reduction_factor) for x in batch]
    in_lens = [len(x) for x in xs]
    out_lens = [len(y) for y in ys]
    in_max_len = max(in_lens)
    out_max_len = max(out_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    y_batch = torch.stack([torch.from_numpy(pad_2d(y, out_max_len)) for y in ys])
    il_batch = torch.tensor(in_lens, dtype=torch.long)
    ol_batch = torch.tensor(out_lens, dtype=torch.long)
    stop_flags = torch.zeros(y_batch.shape[0], y_batch.shape[1])
    for idx, out_len in enumerate(out_lens):
        stop_flags[idx, out_len - 1:] = 1.0
    return x_batch, il_batch, y_batch, ol_batch, stop_flags
