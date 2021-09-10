import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.append("../..")
from vc_tts_template.dsp import logmelspectrogram
from vc_tts_template.frontend.openjtalk import text_to_sequence


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--filter_length", type=int)
    parser.add_argument("--hop_length", type=int)
    parser.add_argument("--win_length", type=int)
    parser.add_argument("--n_mel_channels", type=int)
    parser.add_argument("--mel_fmin", type=int)
    parser.add_argument("--mel_fmax", type=int)
    parser.add_argument("--multi_speaker", type=int)
    parser.add_argument("--pitch_phoneme_averaging", type=int)
    parser.add_argument("--energy_phoneme_averaging", type=int)
    return parser


def get_alignment(tier, sr, hop_length):
    sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append('sp')

        durations.append(
            int(
                np.round(e * sr / hop_length)
                - np.round(s * sr / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    assert len(phones) == len(durations)
    return phones, durations, start_time, end_time


def process_utterance(wav_path, lab_path, sr, n_fft, hop_length, win_length,
                      n_mels, fmin, fmax,
                      pitch_phoneme_averaging, energy_phoneme_averaging):
    textgrid = tgt.io.read_textgrid(lab_path)
    text, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones"), sr, hop_length
    )
    if start >= end:
        return None
    # Read and trim wav files
    wav, _ = librosa.load(wav_path, sr=sr)
    wav = wav[
        int(sr * start): int(sr * end)
    ].astype(np.float32)
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64),
                         pitch, t, sr)
    pitch = pitch[: sum(duration)]
    if np.sum(pitch != 0) <= 1:
        return None
    mel_spectrogram, energy = logmelspectrogram(
        wav,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        need_energy=True
    )
    mel_spectrogram = mel_spectrogram[: sum(duration), :]
    energy = energy[: sum(duration)]

    if pitch_phoneme_averaging is True:
        # perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos: pos + d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[: len(duration)]

    if energy_phoneme_averaging is True:
        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos: pos + d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[: len(duration)]

    return (
        text,
        mel_spectrogram,  # (T, mel)
        pitch,
        energy,
        np.array(duration)
    )


def preprocess(
    wav_file,
    lab_file,
    sr,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    fmin,
    fmax,
    pitch_phoneme_averaging,
    energy_phoneme_averaging,
    in_dir,
    out_dir,
):
    assert wav_file.stem == lab_file.stem

    text, mel_spectrogram, pitch, energy, duration = process_utterance(
        wav_file,
        lab_file,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        pitch_phoneme_averaging,
        energy_phoneme_averaging
    )
    text = np.array(text_to_sequence(text), dtype=np.int64)

    # save to files
    # mel: (T, mel_dim)
    utt_id = lab_file.stem
    np.save(in_dir / f"{utt_id}-feats.npy", text, allow_pickle=False)
    np.save(
        out_dir / "mel" / f"{utt_id}-feats.npy",
        mel_spectrogram.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out_dir / "pitch" / f"{utt_id}-feats.npy",
        pitch.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out_dir / "energy" / f"{utt_id}-feats.npy",
        energy.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        out_dir / "duration" / f"{utt_id}-feats.npy",
        duration.astype(np.int32),
        allow_pickle=False,
    )
    return (
        wav_file.name.split("_")[0],
        np.min(pitch),
        np.max(pitch),
        np.min(energy),
        np.max(energy)
    )


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]

    wav_files = [Path(args.wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_files = [Path(args.lab_root) / f"{utt_id}.TextGrid" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_fastspeech2"
    out_dir = Path(args.out_dir) / "out_fastspeech2"
    out_mel_dir = Path(args.out_dir) / "out_fastspeech2" / "mel"
    out_pitch_dir = Path(args.out_dir) / "out_fastspeech2" / "pitch"
    out_energy_dir = Path(args.out_dir) / "out_fastspeech2" / "energy"
    out_duration_dir = Path(args.out_dir) / "out_fastspeech2" / "duration"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mel_dir.mkdir(parents=True, exist_ok=True)
    out_pitch_dir.mkdir(parents=True, exist_ok=True)
    out_energy_dir.mkdir(parents=True, exist_ok=True)
    out_duration_dir.mkdir(parents=True, exist_ok=True)

    speaker_lst: List[str] = []
    stats_tmp: Dict[str, List[float]] = {"pitch_min": [], "pitch_max": [], "energy_min": [], "energy_max": []}

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                wav_file,
                lab_file,
                args.sample_rate,
                args.filter_length,
                args.hop_length,
                args.win_length,
                args.n_mel_channels,
                args.mel_fmin,
                args.mel_fmax,
                args.pitch_phoneme_averaging > 0,
                args.energy_phoneme_averaging > 0,
                in_dir,
                out_dir,
            )
            for wav_file, lab_file in zip(wav_files, lab_files)
        ]
        for future in tqdm(futures):
            spk, p_m, p_M, e_m, e_M = future.result()
            speaker_lst.append(spk)
            stats_tmp["pitch_min"].append(p_m)
            stats_tmp["pitch_max"].append(p_M)
            stats_tmp["energy_min"].append(e_m)
            stats_tmp["energy_max"].append(e_M)

    if args.multi_speaker > 0:
        speakers_path = Path(args.out_dir).parent / "speakers.json"
        speaker_set = set(speaker_lst)  # type: ignore
        speakers = {}
        if speakers_path.exists():
            with open(speakers_path) as f:
                speakers = json.load(f)
        for spk in speaker_set:
            if spk not in speakers.keys():
                speakers[spk] = len(speakers.keys())

        with open(speakers_path, "w") as f:
            f.write(json.dumps(speakers))

    stats_path = Path(args.out_dir).parent / "stats.json"
    stats = {"pitch_min": 1e+9, "pitch_max": -1e+9, "energy_min": 1e+9, "energy_max": -1e-9}

    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    stats["pitch_min"] = float(min(stats["pitch_min"], np.min(stats_tmp["pitch_min"])))
    stats["pitch_max"] = float(max(stats["pitch_max"], np.max(stats_tmp["pitch_max"])))
    stats["energy_min"] = float(min(stats["energy_min"], np.min(stats_tmp["energy_min"])))
    stats["energy_max"] = float(max(stats["energy_max"], np.max(stats_tmp["energy_max"])))

    with open(stats_path, "w") as f:
        f.write(json.dumps(stats))
