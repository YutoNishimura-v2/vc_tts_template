import numpy as np

from vc_tts_template.utils import pad_2d


def reprocess(batch, idxs, speaker_dict, emotion_dict):
    file_names = [batch[idx][0] for idx in idxs]
    s_mels = [batch[idx][1] for idx in idxs]
    t_mels = [batch[idx][2] for idx in idxs]

    if speaker_dict is not None:
        s_speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in file_names])
        t_speakers = np.array([speaker_dict[fname.split("_")[1]] for fname in file_names])
    else:
        s_speakers = np.array([0 for _ in idxs])
        t_speakers = np.array([0 for _ in idxs])
    if emotion_dict is not None:
        s_emotions = np.array([emotion_dict[fname.split("_")[-2]] for fname in file_names])
        t_emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in file_names])
    else:
        s_emotions = np.array([0 for _ in idxs])
        t_emotions = np.array([0 for _ in idxs])

    s_mel_lens = np.array([s_mel.shape[0] for s_mel in s_mels])
    t_mel_lens = np.array([t_mel.shape[0] for t_mel in t_mels])

    ids = np.array(file_names)
    s_mels = pad_2d(s_mels)
    t_mels = pad_2d(t_mels)

    return (
        ids,
        s_speakers,
        t_speakers,
        s_emotions,
        t_emotions,
        s_mels,
        s_mel_lens,
        max(s_mel_lens),
        t_mels,
        t_mel_lens,
        max(t_mel_lens),
    )


def collate_fn_tacotron2VC(batch, batch_size, speaker_dict=None, emotion_dict=None):
    """Collate function for Tacotron.
    Args:
        batch (list): List of tuples of the form (inputs, targets).
        Datasetのreturnが1単位となって, それがbatch_size分入って渡される.
    Returns:
        tuple: Batch of inputs, input lengths, targets, target lengths and stop flags.
    """
    # shape[0]がtimeになるようなindexを指定する.
    len_arr = np.array([batch[idx][1].shape[0] for idx in range(len(batch))])
    # 以下固定
    idx_arr = np.argsort(-len_arr)
    tail = idx_arr[len(idx_arr) - (len(idx_arr) % batch_size):]
    idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % batch_size)]
    idx_arr = idx_arr.reshape((-1, batch_size)).tolist()
    if len(tail) > 0:
        idx_arr += [tail.tolist()]
    output = list()

    # 以下, reprocessへの引数が変更の余地あり.
    for idx in idx_arr:
        output.append(reprocess(batch, idx, speaker_dict, emotion_dict))

    return output
