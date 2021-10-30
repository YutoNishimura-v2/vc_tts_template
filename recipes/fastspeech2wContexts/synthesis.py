from pathlib import Path
import sys

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile

sys.path.append("../..")
from vc_tts_template.fastspeech2wContexts.gen import synthesis
from vc_tts_template.fastspeech2wContexts.collate_fn import (
    make_dialogue_dict, get_embs
)
from vc_tts_template.utils import load_utt_list, optional_tqdm


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=device,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # vocoder
    vocoder_config = OmegaConf.load(to_absolute_path(config.vocoder.model_yaml))
    vocoder_model = hydra.utils.instantiate(vocoder_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.vocoder.checkpoint),
        map_location=device,
    )
    vocoder_model.load_state_dict(checkpoint["state_dict"]["netG"])
    vocoder_model.eval()
    vocoder_model.remove_weight_norm()

    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # data prepare
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    if config.reverse:
        utt_ids = utt_ids[::-1]

    lab_files = [in_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]

    # read text embeddings
    dialogue_info = to_absolute_path(config.dialogue_info)
    utt2id, id2utt = make_dialogue_dict(dialogue_info)
    text_emb_paths = list(Path(to_absolute_path(config.emb_dir)).glob("*.npy"))
    use_hist_num = config.use_hist_num
    # read prosody embeddings
    prosody_emb_paths = list(Path(to_absolute_path(config.prosody_emb_dir)).glob("*.npy"))
    g_prosody_emb_paths = list(Path(to_absolute_path(config.g_prosody_emb_dir)).glob("*.npy"))

    # prepare text embedding
    context_embeddings = []
    for utt_id in utt_ids:
        current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions = get_embs(
            utt_id, text_emb_paths,
            utt2id, id2utt, use_hist_num
        )
        context_embeddings.append([current_txt_emb, history_txt_embs, hist_emb_len, history_speakers, history_emotions])

    # prepare prosody embedding
    prosody_embeddings = []
    for utt_id in utt_ids:
        if len(prosody_emb_paths) > 0:
            _, history_prosody_emb, _, _, _ = get_embs(
                utt_id, prosody_emb_paths,
                utt2id, id2utt, use_hist_num,
                start_index=1, only_latest=True,
                use_local_prosody_hist_idx=config.use_local_prosody_hist_idx
            )
            _, history_g_prosody_embs, _, _, _ = get_embs(
                utt_id, g_prosody_emb_paths,
                utt2id, id2utt, use_hist_num,
                start_index=1
            )
            prosody_embeddings.append([history_prosody_emb, history_g_prosody_embs])
        else:
            prosody_embeddings.append([None, None])

    if config.num_eval_utts is not None and config.num_eval_utts > 0:
        lab_files = lab_files[: config.num_eval_utts]

    # Run synthesis for each utt.
    for lab_file, context_embedding, prosody_embedding in optional_tqdm(config.tqdm, desc="Utterance")(
        zip(lab_files, context_embeddings, prosody_embeddings)
    ):
        wav = synthesis(
            device, lab_file, context_embedding, prosody_embedding,
            acoustic_config.netG.speakers, acoustic_config.netG.emotions,
            acoustic_model, acoustic_out_scaler, vocoder_model
        )

        wav = np.clip(wav, -1.0, 1.0)

        utt_id = Path(lab_file).name.replace("-feats.npy", "")
        out_wav_path = out_dir / f"{utt_id}.wav"
        wavfile.write(
            out_wav_path,
            rate=config.sample_rate,
            data=(wav * 32767.0).astype(np.int16),
        )

    # add reconstruct wav output
    out_dir = out_dir / "reconstruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.in_mel_dir is not None:
        in_mel_dir = Path(to_absolute_path(config.in_mel_dir))
        mel_files = [in_mel_dir / f"{utt_id.strip()}-feats.npy" for utt_id in utt_ids]

        if config.num_eval_utts is not None and config.num_eval_utts > 0:
            mel_files = mel_files[: config.num_eval_utts]

        for mel_file in optional_tqdm(config.tqdm, desc="Utterance")(mel_files):
            mel_org = np.load(mel_file)
            mel_org = acoustic_out_scaler.inverse_transform(mel_org)  # type: ignore
            mel_org = torch.Tensor(mel_org).unsqueeze(0).to(device)
            wav = vocoder_model(mel_org.transpose(1, 2)).squeeze(1).cpu().data.numpy()[0]
            utt_id = Path(mel_file).name.replace("-feats.npy", "")
            out_wav_path = out_dir / f"{utt_id}.wav"
            wavfile.write(
                out_wav_path,
                rate=config.sample_rate,
                data=(wav * 32767.0).astype(np.int16),
            )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
