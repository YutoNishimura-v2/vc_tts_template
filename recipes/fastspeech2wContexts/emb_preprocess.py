import argparse
import sys

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
    parser.add_argument(
        "--input_duration_paths",
        default="",
        type=init_for_list,
        help='durationのbasepathを「，」区切りで与えてください.'
    )
    parser.add_argument(
        "--output_mel_file_paths",
        default="",
        type=init_for_list,
    )
    parser.add_argument(
        "--model_config_paths",
        default="",
        type=init_for_list,
    )
    parser.add_argument(
        "--pretrained_checkpoints",
        default="",
        type=init_for_list,
    )

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


def get_prosody_embeddings(
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
    if len(input_duration_paths[0]) == 0:
        return

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
                    batch_duration = torch.tensor(batch_duration).to(device)
                    batch_mel = torch.tensor(batch_mel).to(device)

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


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.out_dir)
    in_text_emb_dir = in_dir / "text_emb"

    in_dir.mkdir(parents=True, exist_ok=True)
    in_text_emb_dir.mkdir(parents=True, exist_ok=True)
    print("get text embeddings...")
    get_text_embeddings(args.dialogue_info, in_text_emb_dir, args.BERT_weight)
    print("get prosody embeddings...")
    get_prosody_embeddings(
        args.input_duration_paths, args.output_mel_file_paths,
        args.out_dir, args.model_config_paths, args.pretrained_checkpoints
    )
