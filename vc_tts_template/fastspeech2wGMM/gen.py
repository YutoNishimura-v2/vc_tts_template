import numpy as np
import torch
from pathlib import Path
import tgt
from vc_tts_template.frontend.openjtalk import text_to_sequence


@torch.no_grad()
def synthesis(device, lab_file, speaker_dict, emotion_dict, acoustic_model,
              acoustic_out_scaler, vocoder_model):
    textgrid = tgt.io.read_textgrid(lab_file)
    phones = []
    sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']
    for t in textgrid.get_tier_by_name("phones")._objects:
        p = t.text
        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append('sp')
    # Trim tailing silences
    phones = phones[:end_idx]

    ids = [Path(lab_file).name.replace(".TextGrid", "")]
    if speaker_dict is None:
        speakers = np.array([0])
    else:
        speakers = np.array([speaker_dict[fname.split("_")[0]] for fname in ids])
    if emotion_dict is None:
        emotions = np.array([0])
    else:
        emotions = np.array([emotion_dict[fname.split("_")[-1]] for fname in ids])
    in_feats = np.array(text_to_sequence(phones), dtype=np.int64)
    src_lens = [in_feats.shape[0]]
    max_src_len = max(src_lens)

    speakers = torch.tensor(speakers, dtype=torch.long).to(device)
    emotions = torch.tensor(emotions, dtype=torch.long).to(device)
    in_feats = torch.tensor(in_feats, dtype=torch.long).unsqueeze(0).to(device)
    src_lens = torch.tensor(src_lens, dtype=torch.long).to(device)

    output = acoustic_model(
        ids=ids,
        speakers=speakers,
        emotions=emotions,
        texts=in_feats,
        src_lens=src_lens,
        max_src_len=max_src_len,
    )

    mel_post = output[1]
    mels = [acoustic_out_scaler.inverse_transform(mel.cpu().data.numpy()) for mel in mel_post]  # type: ignore
    mels = torch.Tensor(np.array(mels)).to(device)
    wavs = vocoder_model(mels.transpose(1, 2)).squeeze(1).cpu().data.numpy()

    return wavs[0]
