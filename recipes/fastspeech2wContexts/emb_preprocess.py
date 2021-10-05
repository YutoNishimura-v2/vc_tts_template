import argparse
import sys

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dialogue_info", type=str, help="dialogue_info")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--BERT_weight", type=str)
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
        if ((i+1) % batch_size) or (i == len(file_names)-1):
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


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.out_dir)
    in_text_emb_dir = in_dir / "text_emb"

    in_dir.mkdir(parents=True, exist_ok=True)
    in_text_emb_dir.mkdir(parents=True, exist_ok=True)
    get_text_embeddings(args.dialogue_info, in_text_emb_dir, args.BERT_weight)
