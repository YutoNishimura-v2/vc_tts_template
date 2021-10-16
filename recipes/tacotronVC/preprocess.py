import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment, silence

sys.path.append("../..")
from recipes.fastspeech2VC.utils import pydub_to_np
from tqdm import tqdm
from vc_tts_template.dsp import logmelspectrogram


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for tacotron2VC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("src_wav_root", type=str, help="source wav root")
    parser.add_argument("tgt_wav_root", type=str, help="target wav root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--silence_thresh_h", type=int, help="silence thresh of head")
    parser.add_argument("--silence_thresh_t", type=int, help="silence thresh of tail")
    parser.add_argument("--chunk_size", type=int, help="silence chunk size")
    parser.add_argument("--filter_length", type=int)
    parser.add_argument("--hop_length", type=int)
    parser.add_argument("--win_length", type=int)
    parser.add_argument("--n_mel_channels", type=int)
    parser.add_argument("--mel_fmin", type=int)
    parser.add_argument("--mel_fmax", type=int)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--log_base", type=str)
    return parser


def make_novoice_to_zero(audio: AudioSegment, silence_thresh: float, min_silence_len: int) -> AudioSegment:
    """無音判定をくらった部分を, 0にしてしまう.
    """
    silences = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    audio_new = AudioSegment.empty()
    s_index = 0
    for silence_ in silences:
        audio_new += audio[s_index:silence_[0]]
        audio_new += AudioSegment.silent(duration=silence_[1]-silence_[0])
        s_index = silence_[1]

    audio_new += audio[s_index:]

    return audio_new


def delete_novoice(input_path, silence_thresh_h, silence_thresh_t, chunk_size):
    """無音区間を先頭と末尾から削除します.
    Args:
      input_path: wavファイルへのpath
      output_path: wavファイルを貯めたい場所. ファイル名はinput_pathのbasenameからとる.
      chunk_size: 削除に用いる音声の最小単位. 基本defaultのままで良さそう.
    """
    audio = AudioSegment.from_wav(input_path)
    assert len(audio) > 0, f"{input_path}は音声が入っていないようです"

    # 参考: https://stackoverflow.com/questions/29547218/
    # remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
        trim_ms = 0  # ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    start_trim = detect_leading_silence(audio, silence_threshold=silence_thresh_h, chunk_size=chunk_size)
    end_trim = detect_leading_silence(audio.reverse(),
                                      silence_threshold=silence_thresh_t, chunk_size=chunk_size)

    audio_cut = audio[start_trim:len(audio)-end_trim]

    audio_cut = make_novoice_to_zero(audio_cut, silence_thresh_t, chunk_size)

    assert len(audio_cut) > 0, f"{input_path}はすべてcutされてしまいました. 閾値を下げてください"
    return pydub_to_np(audio_cut)


def preprocess(
    src_wav_file,
    tgt_wav_file,
    sr,
    silence_thresh_h,
    silence_thresh_t,
    chunk_size,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    fmin,
    fmax,
    clip_thresh,
    log_base,
    in_dir,
    out_dir,
):
    assert src_wav_file.stem == tgt_wav_file.stem

    src_wav, sr_src = delete_novoice(src_wav_file, silence_thresh_h, silence_thresh_t, chunk_size)
    tgt_wav, sr_tgt = delete_novoice(tgt_wav_file, silence_thresh_h, silence_thresh_t, chunk_size)

    src_wav = librosa.resample(src_wav, sr_src, sr)
    tgt_wav = librosa.resample(tgt_wav, sr_tgt, sr)

    utt_id = src_wav_file.stem

    src_mel = logmelspectrogram(
        src_wav,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        clip=clip_thresh,
        log_base=log_base,
    )
    tgt_mel = logmelspectrogram(
        tgt_wav,
        sr,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        clip=clip_thresh,
        log_base=log_base,
    )
    np.save(
        in_dir / f"{utt_id}-feats.npy",
        src_mel.astype(np.float32),
        allow_pickle=False
    )
    np.save(
        out_dir / f"{utt_id}-feats.npy",
        tgt_mel.astype(np.float32),
        allow_pickle=False,
    )


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]

    src_wav_files = [Path(args.src_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    tgt_wav_files = [Path(args.tgt_wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_tacotron2VC"
    out_dir = Path(args.out_dir) / "out_tacotron2VC"

    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                src_wav_file,
                tgt_wav_file,
                args.sample_rate,
                args.silence_thresh_h,
                args.silence_thresh_t,
                args.chunk_size,
                args.filter_length,
                args.hop_length,
                args.win_length,
                args.n_mel_channels,
                args.mel_fmin,
                args.mel_fmax,
                args.clip,
                args.log_base,
                in_dir,
                out_dir,
            )
            for src_wav_file, tgt_wav_file in zip(src_wav_files, tgt_wav_files)
        ]
        for future in tqdm(futures):
            future.result()
