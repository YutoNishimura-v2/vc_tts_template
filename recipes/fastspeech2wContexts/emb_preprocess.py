import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

sys.path.append("../..")
from vc_tts_template.utils import adaptive_load_state_dict, pad_1d, pad_2d
from recipes.fastspeech2wContexts.preprocess import process_utterance


def init_for_list(input_):
    return input_.split(',')


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for wContexts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dialogue_info", type=str, help="dialogue_info")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--BERT_weight", type=str)
    # for GMM
    parser.add_argument(
        "--input_duration_paths",
        type=init_for_list,
        help='durationのbasepathを「，」区切りで与えてください.',
        nargs='?'
    )
    parser.add_argument("--output_mel_file_paths", type=init_for_list, nargs='?')
    parser.add_argument("--model_config_paths", type=init_for_list, nargs='?')
    parser.add_argument("--pretrained_checkpoints", type=init_for_list, nargs='?')
    # for wav
    parser.add_argument("--input_wav_paths", type=init_for_list, nargs='?')
    parser.add_argument("--input_lab_paths", type=init_for_list, nargs='?')
    parser.add_argument("--input_textgrid_paths", type=init_for_list, nargs='?')
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--filter_length", type=int)
    parser.add_argument("--hop_length", type=int)
    parser.add_argument("--win_length", type=int)
    parser.add_argument("--n_mel_channels", type=int)
    parser.add_argument("--mel_fmin", type=int)
    parser.add_argument("--mel_fmax", type=int)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--log_base", type=str)
    parser.add_argument("--pitch_phoneme_averaging", type=int)
    parser.add_argument("--energy_phoneme_averaging", type=int)
    parser.add_argument("--n_jobs", type=int)
    return parser


def get_text_embeddings(input_txt_file_path, output_dir, BERT_weight, batch_size=16):
    model = SentenceTransformer(BERT_weight)

    with open(input_txt_file_path, 'r') as f:
        texts = f.readlines()
    file_names = [text.strip().split(':')[0] for text in texts]
    texts = [text.strip().split(':')[-1] for text in texts]

    batch_id = []
    batch_text = []
    for i, (id_, text) in tqdm(enumerate(zip(file_names, texts))):
        batch_id.append(id_)
        batch_text.append(text)
        if ((i+1) % batch_size == 0) or (i == len(file_names)-1):
            output = model.encode(batch_text)
            for filename, text_emb in zip(batch_id, output):
                np.save(
                    output_dir / f"{filename}-feats.npy",
                    text_emb.astype(np.float32),
                    allow_pickle=False,
                )
            batch_id = []
            batch_text = []
        else:
            continue


def get_prosody_embeddings_wGMM(
    input_duration_paths, output_mel_file_paths,
    output_dir,
    model_config_paths, pretrained_checkpoints,
    batch_size=16
):
    """対応するデータフォルダと, modelのpretrainを複数指定する.
    想定:
      dump/spk_sr22050/train/in_*をいくつか指定する.
      また, 既に正規化されている前提.
    """
    output_dir = Path(output_dir)
    (output_dir / "prosody_emb").mkdir(parents=True, exist_ok=True)
    (output_dir / "g_prosody_emb").mkdir(parents=True, exist_ok=True)
    # model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for (
        input_duration_path_base, output_mel_file_path_base,
        model_config_path, pretrained_checkpoint
    ) in zip(
        input_duration_paths, output_mel_file_paths,
        model_config_paths, pretrained_checkpoints
    ):
        print(f"process {output_mel_file_path_base}...")
        # model init
        checkpoint = torch.load(to_absolute_path(pretrained_checkpoint), map_location=device)
        model_config = OmegaConf.load(to_absolute_path(model_config_path))
        model = hydra.utils.instantiate(model_config.netG).to(device)
        adaptive_load_state_dict(model, checkpoint["state_dict"])
        model.eval()
        assert model.global_prosody, "this is allowed only for the model which use global prosody"
        # data setting
        for s in ["train", "dev", "eval"]:
            input_duration_path = Path(input_duration_path_base.format(s))
            output_mel_file_path = Path(output_mel_file_path_base.format(s))

            batch_id = []
            batch_src_len = []
            batch_duration = []
            batch_mel = []
            input_duration_paths = list(input_duration_path.glob("*.npy"))
            for i, input_duration_path_ in tqdm(enumerate(input_duration_paths)):
                filename = input_duration_path_.name
                utt_id = filename.replace("-feats.npy", "")
                input_duration = np.load(input_duration_path_)
                output_mel = np.load(output_mel_file_path / filename)
                batch_id.append(utt_id)
                batch_src_len.append(len(input_duration))
                batch_duration.append(input_duration)
                batch_mel.append(output_mel)
                if ((i+1) % batch_size == 0) or (i == len(input_duration_paths)-1):
                    # prepare start
                    sort_idx = np.argsort(-np.array(batch_src_len))
                    batch_id = [batch_id[idx] for idx in sort_idx]
                    batch_src_len = [batch_src_len[idx] for idx in sort_idx]
                    batch_duration = [batch_duration[idx] for idx in sort_idx]
                    batch_mel = [batch_mel[idx] for idx in sort_idx]

                    batch_duration = pad_1d(batch_duration)
                    batch_mel = pad_2d(batch_mel)

                    batch_src_len = torch.tensor(batch_src_len, dtype=torch.long).to(device)
                    batch_duration = torch.tensor(batch_duration, dtype=torch.long).to(device)
                    batch_mel = torch.tensor(batch_mel, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        prosody_target, g_prosody_target = model.prosody_extractor(
                            batch_mel, batch_duration, batch_src_len
                        )
                    for filename, prosody_emb, g_prosody_emb in zip(batch_id, prosody_target, g_prosody_target):
                        np.save(
                            output_dir / "prosody_emb" / f"{filename}-feats.npy",
                            prosody_emb.cpu().numpy().astype(np.float32),
                            allow_pickle=False,
                        )
                        np.save(
                            output_dir / "g_prosody_emb" / f"{filename}-feats.npy",
                            g_prosody_emb.cpu().numpy().astype(np.float32),
                            allow_pickle=False,
                        )

                    batch_id = []
                    batch_src_len = []
                    batch_duration = []
                    batch_mel = []
                else:
                    continue


def get_prosody_embeddings_wWav(
    input_wav_paths, input_lab_paths, input_textgrid_paths, output_dir,
    sample_rate, filter_length, hop_length, win_length,
    n_mel_channels, mel_fmin, mel_fmax, clip, log_base,
    pitch_phoneme_averaging, energy_phoneme_averaging, n_jobs
):
    if (input_lab_paths is not None) and (input_textgrid_paths is not None):
        raise ValueError("labを利用したいのか, textgridを利用したいのか, どちらか一方のみを埋めてください")

    input_context_path = input_lab_paths if input_lab_paths is not None else input_textgrid_paths
    input_context_path_postfix = ".lab" if input_lab_paths is not None else ".TextGrid"

    output_dir = Path(output_dir)
    output_dir_prosody = output_dir / "prosody_emb"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_prosody.mkdir(parents=True, exist_ok=True)

    utt_ids = []

    for input_wav_path_base, input_context_path_base in zip(input_wav_paths, input_context_path):
        wav_files = list(Path(input_wav_path_base).glob("*.wav"))
        lab_files = [
            Path(input_context_path_base) / (input_wav_path.stem + input_context_path_postfix)
            for input_wav_path in wav_files
        ]
        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(
                    process_utterance,
                    wav_file,
                    lab_file,
                    sample_rate,
                    filter_length,
                    hop_length,
                    win_length,
                    n_mel_channels,
                    mel_fmin,
                    mel_fmax,
                    clip,
                    log_base,
                    pitch_phoneme_averaging > 0,
                    energy_phoneme_averaging > 0,
                    input_context_path_postfix == ".lab",
                    return_utt_id=True
                )
                for wav_file, lab_file in zip(wav_files, lab_files)
            ]
            for future in tqdm(futures):
                _, _, pitch, energy, _, utt_id = future.result()
                prosody = np.stack([pitch, energy], -1)
                np.save(
                    output_dir_prosody / f"{utt_id}-feats.npy",
                    prosody.astype(np.float32),
                    allow_pickle=False,
                )
                utt_ids.append(utt_id+'\n')
    with open(output_dir / "prosody_emb.list", "w") as f:
        f.writelines(utt_ids)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.out_dir)
    in_text_emb_dir = in_dir / "text_emb"

    in_dir.mkdir(parents=True, exist_ok=True)
    in_text_emb_dir.mkdir(parents=True, exist_ok=True)
    print("get text embeddings...")
    get_text_embeddings(args.dialogue_info, in_text_emb_dir, args.BERT_weight)

    if (args.input_duration_paths is not None) and (args.input_wav_paths is not None):
        raise ValueError("GMMを利用したいのか, wavを利用したいのか, どちらか一方のみを埋めてください")

    if args.input_duration_paths is not None:
        print("get prosody embeddings from GMM...")
        get_prosody_embeddings_wGMM(
            args.input_duration_paths, args.output_mel_file_paths,
            args.out_dir, args.model_config_paths, args.pretrained_checkpoints
        )
    elif args.input_wav_paths is not None:
        print("get prosody embeddings from wavs...")
        get_prosody_embeddings_wWav(
            args.input_wav_paths, args.input_lab_paths, args.input_textgrid_paths, args.out_dir,
            args.sample_rate, args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.mel_fmin, args.mel_fmax, args.clip, args.log_base,
            args.pitch_phoneme_averaging, args.energy_phoneme_averaging,
            args.n_jobs,
        )
