# fastspeech2利用方法.
## 事前準備
1. 利用したいコーパスのwavをどこかに置く.
    - srは変更する必要なし.
    - `./run.sh`を実行する場所からの相対パスを指定する.
    - 例: "../../../dataset/out_JSUT/JSUT/wav"
2. 利用したいコーパスのtextgridをどこかに置く.
    - 例: "../../../dataset/out_JSUT/textgrid"
    - textgridのファイルを基準にファイルリストを作成します.
3. 終わり. ./run.shを実行していってください.

**注意**
- multi-speakerにしたい場合は, ファイル名の前に話者名を「_」を付けて追加すること.
    - 例: `JSUT_BASIC5000_0001.wav`
    - そして, netGのspeakersに事前に用意したspeakerのdictionalyを追加すること.
- emotion embを付けたい場合は, ファイル名の**後**に感情を「_」をつけて追加.
    - 例: `JSUT_BASIC5000_0001_悲しみ.wav`
    - そして, fastspeech2wEmotionのnetGのemotionsに事前に用意したspeakerのdictionalyを追加すること.

- statsに関しては, dump以下に全データ通してのそれをjsonとして吐きだすので, それを参考に入力. 

- train/dev/eval splitは毎回shuffleしている. 実験を完全再現したいのであれば, stage 0は実行せず, gitに入っているものをそのまま利用すること.

- 実験名管理について
    - spk: 全体の名前. preprocessed_data名にもなるため, データそのものの変更をするならここを変える.
    - tag: 毎度おなじみexp_name. 実験名の変更です. 訓練時.

## 2021-09-29: fastspeech2wGMM追加
### 仕様
- devのlossは, 正解melは与えて, pitch, energyを与えないで計算されているが, 正解melはloss計算用の正解prosody embを計算するためだけに用いられて, 出力のmel自体はduration以外inference時と同じ設定で計算されている.

### 使い方
- configにおいて, 以下のモデル名を指定
    - fastspeech2wGMM
- train_fastspeech2 configにおいて, 以下のlossを指定
    - vc_tts_template.fastspeech2wGMM.loss.FastSpeech2Loss
    - betaを決定
- 後は同様
