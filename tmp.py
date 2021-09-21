from pathlib import Path
from scipy.io import wavfile
import pyworld as pw
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


wav_path_src = Path("recipes/fastspeech2VC/downloads/jsut_jsss_jvs/source")
wav_path_tgt = Path("recipes/fastspeech2VC/downloads/jsut_jsss_jvs/target")
hop_length = 256
n_jobs = 32

def preprocess(wav_path):
    sr, wav = wavfile.read(wav_path)
    pitch, t = pw.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64),
                        pitch, t, sr)
    if np.sum(pitch != 0) <= 1:
        print(wav_path)


with ProcessPoolExecutor(n_jobs) as executor:
    futures = [
        executor.submit(
            preprocess,
            wav_path
        )
        for wav_path in list(wav_path_src.glob("*.wav")) + list(wav_path_tgt.glob("*.wav"))
    ]
    for future in tqdm(futures):
        future.result()
