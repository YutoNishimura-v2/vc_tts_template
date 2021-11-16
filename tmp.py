import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

all_utt_id = []
text_emb = Path("recipes/fastspeech2wContexts/dump/LINE_wContext_2_sr22050/text_emb")

for new_path_id in text_emb.glob("*.npy"):
    all_utt_id.append(new_path_id.name)

g_prosody_emb_base = Path("recipes/fastspeech2wContexts/dump/LINE_wContextwProsody_sr22050/g_prosody_emb")
prosody_emb_base = Path("recipes/fastspeech2wContexts/dump/LINE_wContextwProsody_sr22050/prosody_emb")

out_g_prosody_emb_base = Path("recipes/fastspeech2wContexts/dump/LINE_wContextwProsody_2_sr22050/g_prosody_emb")
out_prosody_emb_base = Path("recipes/fastspeech2wContexts/dump/LINE_wContextwProsody_2_sr22050/prosody_emb")

out_g_prosody_emb_base.mkdir(exist_ok=True)
out_prosody_emb_base.mkdir(exist_ok=True)

p_embs = list(prosody_emb_base.glob("*.npy"))

def copy_emb(p_emb_path):
    g_emb_path = g_prosody_emb_base / p_emb_path.name

    utt_id_woEmo = p_emb_path.stem.replace("_"+p_emb_path.stem.split("_")[-1], "")
    new_filename = ""
    for utt_id in all_utt_id:
        if utt_id_woEmo in utt_id:
            new_filename = utt_id
            break

    shutil.copy(p_emb_path, out_prosody_emb_base / new_filename)
    shutil.copy(g_emb_path, out_g_prosody_emb_base / new_filename)


with ProcessPoolExecutor(20) as executor:
    futures = [
        executor.submit(
            copy_emb,
            p_emb,
        )
        for p_emb in p_embs
    ]
    for future in tqdm(futures):
        future.result()
