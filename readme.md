# 共感的対話音声合成実験用コード

## 謝辞
こちらのコードは、[ttslearn](https://github.com/r9y9/ttslearn)をベースにして作られています。hifiganに関しては、[こちら](https://github.com/jik876/hifi-gan)、fastspeech2に関しては、[こちら](https://github.com/ming024/FastSpeech2)がベースです。

## 使い方(基本ttslearnと同じです)
- `cd recipes/fastspeech2wContexts` をする
- そこに入っている，3つの config をいじる
    - config.yaml
    - conf/train_fastspeech2/config.yaml
    - conf/train_fastspeech2/model/利用するモデル名.yaml
- `run.sh --stage 番号 --stop-stage 番号`
    - 基本的に番号は「1, 2, 4, 5」を使うと思います

### configの説明
見ただけでは何を設定できるのか分からないものを書きます．
#### config.yaml
- spk: データセット名です
- tag: exp名です
- wav_root: .wavが入ったファイルパスを入れてください
- lab_root: .labが入ったファイルパスを入れてください
- dialogue_info: dialogue_info.txt のパスをいれてください

- train_num: データセットを作成する際にsplitをするので数を入れてください
- accent_info: アクセントを使う(labがfullcontextの場合)は1にしてください
- speakers: 「合成対象音声が」multi-speakerの場合，話者ごとに正規化を行うので話者を記載してください
- use_prosody_hist_num: textのhist_numとprosodyのhist_numを別々にしたい場合，こちらに数値を入れたらprosody のhist_numを別に設定できます
- use_situation_text: 1でsituation text を利用するモードです(dialogue_info.txtにないと無理です)
- use_local_prosody_hist_idx: 最新モデルでは利用していないパラメータです
- input_duration_paths: ここから「pretrained_checkpoints」までもいらないです
- input_wav_paths: prosody embを作成するのに使うwavの入ったパスを入れてください．
- emb_speakers: embeddingに複数話者が登場する場合，話者ごとに正規化をするので話者名を書いてください
- mel_mode: SSLモデルを用いない場合，メルを用いるので「1」，SSLの場合は用いないので「0」にしてください
- pau_split: fine-grained にしたい場合は「1」そうでないなら「0」にしてください
