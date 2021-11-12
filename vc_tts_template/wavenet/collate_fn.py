import numpy as np
import torch


def collate_fn_wavenet(batch, max_time_frames=100, hop_size=80, aux_context_window=2):
    """Collate function for WaveNet.

    Args:
        batch (list): List of tuples of the form (inputs, targets).
        max_time_frames (int, optional): Number of time frames. Defaults to 100.
        hop_size (int, optional): Hop size. Defaults to 80.
        aux_context_window (int, optional): Auxiliary context window. Defaults to 2.

    Returns:
        tuple: Batch of waveforms and conditional features.
    """
    # (注意: 単位をframeで考えるべし. つまり, 最初から全て*srされているものと考える.
    # 全てにかけられているので, ms単位で考えても論理は同じ)
    # hop_sizeはフレームシフト. 要するに, サンプル→フレームに圧縮する時の比になっている.
    # なので, 逆にフレームをサンプル単位に変えたければ, ↓のように, hop_sizeをかければよい
    max_time_steps = max_time_frames * hop_size  # sample単位のtime step.

    xs, cs = [b[1] for b in batch], [b[0] for b in batch]

    # 条件付け特徴量の開始位置をランダム抽出した後、それに相当する短い音声波形を切り出します
    # windowを考慮して選ばないとout of index
    # あと, これだと滅茶苦茶短い音声を入れてしまうとout of indexになる.
    c_lengths = [len(c) for c in cs]
    start_frames = np.array(
        [
            np.random.randint(
                aux_context_window, cl - aux_context_window - max_time_frames
            )
            for cl in c_lengths
        ]
    )
    # サンプル単位に変換
    x_starts = start_frames * hop_size
    x_ends = x_starts + max_time_steps
    # cはフレーム単位なのでそのまま.
    c_starts = start_frames - aux_context_window
    c_ends = start_frames + max_time_frames + aux_context_window
    x_cut = [x[s:e] for x, s, e in zip(xs, x_starts, x_ends)]
    c_cut = [c[s:e] for c, s, e in zip(cs, c_starts, c_ends)]

    # numpy.ndarray のリスト型から torch.Tensor 型に変換します
    x_batch = torch.tensor(x_cut, dtype=torch.long)  # (B, T)
    c_batch = torch.tensor(c_cut, dtype=torch.float).transpose(2, 1)  # (B, C, T')

    return x_batch, c_batch
