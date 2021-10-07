from typing import List, Dict, Callable
from pathlib import Path
from hydra.utils import to_absolute_path

from torch.utils import data as data_utils
import numpy as np

from vc_tts_template.utils import load_utt_list, pad_1d, pad_2d


def make_dialogue_dict(dialogue_info):
    # utt_idを投げたら, 対話IDと対話内IDを返してくれるような辞書と,
    # その逆で, 対話IDと対話内IDを投げたらそのutt_idを返してくれるような辞書を用意.
    utt2id = {}
    id2utt = {}

    with open(dialogue_info, 'r') as f:
        dialogue_data = f.readlines()

    for dialogue in dialogue_data:
        utt_id, dialogue_id, in_dialogue_id, _ = dialogue.strip().split(":")
        utt2id[utt_id] = (dialogue_id, in_dialogue_id)
        id2utt[(dialogue_id, in_dialogue_id)] = utt_id

    return utt2id, id2utt


class fastspeech2_Dataset(data_utils.Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths: List of paths to input files
        out_paths: List of paths to output files
    """

    def __init__(
        self,
        in_paths: List,
        out_mel_paths: List,
        out_pitch_paths: List,
        out_energy_paths: List,
        out_duration_paths: List,
        dialogue_info: Path,
        text_emb_paths: List,
        use_hist_num: int,
    ):
        self.in_paths = in_paths
        self.out_mel_paths = out_mel_paths
        self.out_pitch_paths = out_pitch_paths
        self.out_energy_paths = out_energy_paths
        self.out_duration_paths = out_duration_paths
        self.utt2id, self.id2utt = make_dialogue_dict(dialogue_info)
        self.text_emb_paths = text_emb_paths
        self.use_hist_num = use_hist_num

    def __getitem__(self, idx: int):
        """Get a pair of input and target

        Args:
            idx: index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions = self.get_embs(
            self.in_paths[idx].name.replace("-feats.npy", ""), self.text_emb_paths
        )
        return (
            self.in_paths[idx].name,
            np.load(self.in_paths[idx]),
            np.load(self.out_mel_paths[idx]),
            np.load(self.out_pitch_paths[idx]),
            np.load(self.out_energy_paths[idx]),
            np.load(self.out_duration_paths[idx]),
            current_txt_emb,
            history_txt_embs,
            hist_emb_len,
            history_speakers,
            history_emotions,
        )

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.in_paths)

    def get_embs(self, utt_id, emb_paths):
        current_d_id, current_in_d_id = self.utt2id[utt_id]
        current_emb = np.load(self.get_path_from_uttid(utt_id, emb_paths))

        range_ = range(int(current_in_d_id)-1, max(0, int(current_in_d_id)-1-self.use_hist_num), -1)
        hist_embs = []
        hist_emb_len = 0
        history_speakers = []
        history_emotions = []
        for hist_in_d_id in range_:
            utt_id = self.id2utt[(current_d_id, str(hist_in_d_id))]
            hist_embs.append(np.load(self.get_path_from_uttid(utt_id, emb_paths)))
            history_speakers.append(utt_id.split('_')[0])
            history_emotions.append(utt_id.split('_')[-1])
            hist_emb_len += 1

        for _ in range(self.use_hist_num - len(hist_embs)):
            hist_embs.append(np.zeros_like(current_emb))
            history_speakers.append("pad")
            history_emotions.append("pad")
        return (
            np.array(current_emb), np.stack(hist_embs), hist_emb_len,
            np.array(history_speakers), np.array(history_emotions)
        )

    def get_path_from_uttid(self, utt_id, emb_paths):
        answer = None
        for path_ in emb_paths:
            answer = path_
            if utt_id in path_.name:
                break
        return answer


def fastspeech2_get_data_loaders(data_config: Dict, collate_fn: Callable) -> Dict[str, data_utils.DataLoader]:
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

        emb_dir = Path(to_absolute_path(data_config.emb_dir))  # type:ignore
        dialogue_info = Path(to_absolute_path(data_config.dialogue_info))  # type:ignore

        in_feats_paths = [in_dir / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_mel_paths = [out_dir / "mel" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_pitch_paths = [out_dir / "pitch" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_energy_paths = [out_dir / "energy" / f"{utt_id}-feats.npy" for utt_id in utt_ids]
        out_duration_paths = [out_dir / "duration" / f"{utt_id}-feats.npy" for utt_id in utt_ids]

        text_emb_paths = list((emb_dir / "text_emb").glob("*.npy"))

        dataset = fastspeech2_Dataset(
            in_feats_paths,
            out_mel_paths,
            out_pitch_paths,
            out_energy_paths,
            out_duration_paths,
            dialogue_info,
            text_emb_paths,
            use_hist_num=data_config.use_hist_num  # type:ignore
        )
        data_loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=data_config.batch_size * data_config.group_size,  # type: ignore
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,  # type: ignore
            shuffle=phase.startswith("train"),  # trainならTrue
        )

    return data_loaders


def reprocess(batch, idxs, speaker_dict, emotion_dict):
    file_names = [batch[idx][0] for idx in idxs]
    texts = [batch[idx][1] for idx in idxs]
    mels = [batch[idx][2] for idx in idxs]
    pitches = [batch[idx][3] for idx in idxs]
    energies = [batch[idx][4] for idx in idxs]
    durations = [batch[idx][5] for idx in idxs]
    c_txt_embs = [batch[idx][6] for idx in idxs]
    h_txt_embs = [batch[idx][7] for idx in idxs]
    h_txt_emb_lens = [batch[idx][8] for idx in idxs]
    h_speakers = [batch[idx][9] for idx in idxs]
    h_emotions = [batch[idx][10] for idx in idxs]

    ids = np.array([fname.replace("-feats.npy", "") for fname in file_names])
    if speaker_dict is not None:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
        h_speakers = np.array([[speaker_dict[spk] for spk in speakers] for speakers in h_speakers])
    else:
        raise ValueError("You Need emotion_dict")
    if emotion_dict is not None:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
        h_emotions = np.array([[emotion_dict[emo] for emo in emotions] for emotions in h_emotions])
    else:
        emotions = np.array([-1 for _ in idxs])
        h_emotions = np.array([[-1 for _ in range(len(h_speakers[0]))] for _ in idxs])

    # reprocessの内容をここに.

    text_lens = np.array([text.shape[0] for text in texts])
    mel_lens = np.array([mel.shape[0] for mel in mels])

    texts = pad_1d(texts)
    mels = pad_2d(mels)
    pitches = pad_1d(pitches)
    energies = pad_1d(energies)
    durations = pad_1d(durations)
    c_txt_embs = np.array(c_txt_embs)

    return (
        ids,
        speakers,
        emotions,
        texts,
        text_lens,
        max(text_lens),
        mels,
        mel_lens,
        max(mel_lens),
        pitches,
        energies,
        durations,
        c_txt_embs,
        np.array(h_txt_embs),
        np.array(h_txt_emb_lens),
        h_speakers,
        h_emotions,
    )


def collate_fn_fastspeech2(batch, batch_size, speaker_dict=None, emotion_dict=None):
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
