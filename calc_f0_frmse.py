import numpy as np
import pandas as pd
import pyopenjtalk
import os
import itertools
import glob
from pathlib import Path
import scipy
import librosa
import pyworld as pw
import numpy as np
import librosa
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
import os
from fastdtw import fastdtw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


gt_wav_folder_base = "/home/ynishimura/workspace/python/dataset/LINE"

method_base_pathes = [
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_128_JT_TMCCE_GRU/synthesis_fastspeech2wContextswPEProsody_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_156_Attn/synthesis_fastspeech2wContextswPEProsody_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_14_sr22050_LINE_wContextwPEProsody_157_seg/synthesis_fastspeech2wContextswPEProsody_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_4_sr22050_LINE_wContextwPEProsody_158_SM_woSSL/synthesis_fastspeech2wContextswPEProsody_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_159_SM_wSSL/synthesis_fastspeech2wContextswPEProsody_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_160_Meltarget/synthesis_fastspeech2wContextswPEProsodywCurrentMel_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_162_JT_CMCCE_Attn_Meltarget_MI/synthesis_fastspeech2wContextswPEProsodywCurrentMel_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_163_JT_CMCCE_Attn_Meltarget_MI/synthesis_fastspeech2wContextswPEProsodywCurrentMel_hifigan/eval/real_dialogue_synthesis",
    "recipes/fastspeech2wContexts/exp/LINE_wContextwPEProsody_11_sr22050_LINE_wContextwPEProsody_164_JT_CMCCE_Attn_Meltarget_MI/synthesis_fastspeech2wContextswPEProsodywCurrentMel_hifigan/eval/real_dialogue_synthesis"
]


def get_mgc_f0_for_gt(wavfile, sampling_rate):
    wav, _ = librosa.load(wavfile, sr=sampling_rate)
    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        f0_floor=120,
        f0_ceil=700,
        frame_period=5,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)
    f0 = pitch[pitch != 0]
    spectrogram = pw.cheaptrick(wav.astype(np.float64), pitch, t, sampling_rate)
    mgc = pysptk.sp2mc(spectrogram, 39, alpha)[pitch != 0]

    basename = wavfile.stem

    return basename, mgc, f0


def get_msds_f0_rmse(wavfile, sampling_rate, mgcdict_gt, f0dict_gt):
    wav, _ = librosa.load(wavfile, sr=sampling_rate)
    # Compute fundamental frequency
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sampling_rate,
        f0_floor=120,
        f0_ceil=700,
        frame_period=5,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)
    f0 = pitch[pitch != 0]
    spectrogram = pw.cheaptrick(wav.astype(np.float64), pitch, t, sampling_rate)
    mgc = pysptk.sp2mc(spectrogram, 39, alpha)[pitch != 0]

    basename = wavfile.stem
    # speaker name と emotion name は除去
    basename = "".join(basename.split("_")[1:-1])
    mgc_gt = mgcdict_gt[basename]
    f0_gt = f0dict_gt[basename]
    distance, path = fastdtw(mgc, mgc_gt, dist=euclidean)

    distance /= (len(mgc) + len(mgc_gt))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    mgc, mgc_gt = mgc[pathx], mgc_gt[pathy]
    f0, f0_gt = f0[pathx], f0_gt[pathy]

    frames = mgc.shape[0]

    z = mgc - mgc_gt
    mcd = _logdb_const * np.sqrt((z * z).sum(-1).sum() / frames / 2)

    z = f0 - f0_gt
    f0_rms = np.sqrt((z * z).sum() / frames)

    return mcd, f0_rms


alpha = pysptk.util.mcepalpha(22050)

mgcdict_gt = {}
f0dict_gt = {}
wavfiles = sorted(list(Path(gt_wav_folder_base).glob("**/*.wav")))
sampling_rate = 22050

with ProcessPoolExecutor(30) as executor:
    futures = [
        executor.submit(
            get_mgc_f0_for_gt,
            wavfile,
            sampling_rate,
        )
        for wavfile in wavfiles
    ]
    for future in tqdm(futures):
        basename, mgc, f0 = future.result()
        mgcdict_gt[basename] = mgc
        f0dict_gt[basename] = f0

_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
for method_base in method_base_pathes:
    wavfiles = sorted(list(Path(method_base).glob("**/*.wav")))
    mcds = []
    f0_rmses = []

    with ProcessPoolExecutor(30) as executor:
        futures = [
            executor.submit(
                get_msds_f0_rmse,
                wavfile,
                sampling_rate,
                mgcdict_gt,
                f0dict_gt,
            )
            for wavfile in wavfiles
        ]
        for future in tqdm(futures):
            mcd, f0_rms = future.result()
            mcds.append(mcd)
            f0_rmses.append(f0_rms)

    print(method_base)
    print(" MCD     [dB]: %f += %f" % (np.mean(mcds), np.std(mcds)))
    print(" F0 RMSE [Hz]: %f += %f" % (np.mean(f0_rmses), np.std(f0_rmses)))
