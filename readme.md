# モデルコード一般化プロジェクト
- 目的: 今後, いろんなモデルを実装していくにあたり, 毎回dataloaderとかtrainとか書くのあほくさい.
- なので, 使いまわし可能なコードを作ろう!!!

## run.shの流れ

- stage -1
    - コーパスのダウンロード.
- stage 0
    - dataをtrain, val, evalに分割している.
- stage 1
    - duration modelに対する特徴量作成.
- stage 2
    - acoustic modelに対する特徴量作成.
- stage 3
    - 正規化
- stage 4
    - durationの訓練
- stage 5
    - acousticの訓練
- stage 6
    - synth

## 気になってるところ
- configの基準
    - 2重に書いている変数とかあるけど...
    - ほかのモデルも見て考える.

## 部品に関する感想メモ
### dnnのみ
- recipes: 実行系. confとか出力とかtrainとかも.
    - common: 共通
        - yaml_parser.sh
        - parse_options.sh
            - この2つはstage -1から使用. run.shの基本.
        - qst1.hed @ stage 1
            - 言語特徴に投げるquestion.
            - 使い回し可能だが, そもそも言語特徴量使わなさそう(
        - fit_scaler.py @ stage 3
            - normalizeして, standard scalerを保存するコード.
            - 特徴量の保存ファイル名だけハードコードされているが, それ以外は使いまわせそう.
        - preprocess_normalize.py @ stage 3
            - scalerを渡して, 実際にファイルたちを正規化して保存するコード.
            - これもファイル名だけ決まっているが, 使いまわせそう.
    - dnntts
        - train_dnntts.py @ stage 4
            - train_step: これは完全に専用. lossも中で作っちゃってるし.
            - train_loop: dataloader周りも専用コード.
        - run.sh
            - 実験を再現するファイル. ↓のconfigを利用.
        - config.yaml
            - 実験全般のconfig. runに関して.
            - 一部, train_dnnttsとかぶる内容はあり.
            - その際はrun.sh内で上書きしている.
            - どういう基準でconfig書いているのかいまいち読めない...。
        - preprocess_duration.py
            - 単なる前処理. 使いまわしは困難 @ stage 1
        - preprocess_acoustic.py
            - 同様 @ stage 2
        - synthesis.py @ stage 6
            - これも思いっきりまさにdnntts用. 使えるところはなさそう.
        - data @ stage 0
            - utt_list.txt
                - 使う順にファイル名を書いておいたもの. これはあらかじめ用意するのはそれはそう.
                - gitに配置するべきもの. 実験の再現性確保のため.
        - dump @ stage 2
            - preprocessed dataと同じだと思う.

- vc_tts_template
    - logger.py: 問題なし.
    - utils.py: 問題なし.
    - train_utils.py
        - Dataset: 完全にdnntts専用. ここはでも仕方ないかも.
        - get_data_loaders: これも一部. これもまぁ仕方なし.
        - collate_fn_dnntts: これも専用関数.
        - setup: configに最高に依存している. 少しでも形式変えるとout(それは上の関数も).
    - dnntts
        - model.py: まぁ普通のmodel. 正確にはmoduleしか入ってない.
        - tts.py: inferenceといったほうが近そう. ほぼすべて特徴量の名前とかハードコード. とても使いまわせない.
        - gen.py: これも完全オリジナル.
        - multistream.py: 同様.

#### 感想
- いや, 依存しすぎ. dnnttsという結構deepのようでdeepじゃない手法なので, 複雑なのも仕方ないが...。
- 次のwavenet次第.

### wavenet
dnnttsとの違いに注目して見ていく. 実行に絡んだもののみ記載していく。

- recipes
    - common
        - fit_scaler.py: dnnttsから引き続き. @ stage 4
        今回, wavenetのoutには利用しないが, それはrun.sh内で書く. かしこい.
        フォーマットを統一するからこそ可能な正規化の使いまわし.
        - preprocess_normalize.py: 上に同様.
    - wavenet
        - run.sh: やはり最初らへんは共通. テンプレとして使えそう.
        stage -1, 0 までは完全一致.
        - config.yaml: run.shに必要. 相違点は?
        そもそもwavenetをシステムとしてみた場合, dnnttsとほぼ同じなので, 構成も確かに同じだった. 基準はまだよくわかっていない...少なくとも、「ハイパーパラメータはない」というのは確実。
        実験で頻繁に変えるとしたら↑これだしね。

        - conf: @ stage 5, 6, 7, 8
        おそらく? ここに含まれるconfはすべてtrain/model関係のハイパラ入りconfing+./run.shから渡したいものは渡せるように空白として配置という感じ. そして特にmodelは切り替えられるようにしてある. 分離.
        fastspeech2の実装でいう, preprocess.yaml+その他実行環境については./run.sh管轄のconfig.yamlにいるという感じかもしれない.

        - preprocess_duration.py: dnnttsと同一. @ stage 1
        - preprocess_logf0.py: これも固有 @ stage 2
            思ったのが, ファイル名は全て「ファイル名-feats.npy」で統一していて, こういったpreprocessed_dataを溜めるのは,  dump以下固定, みたいな感じ. この考え方は使えそう.
        - preprocess_wavenet.py: これも同様. @ stage 3
            モデル構造にかかわるような依存も書いている(割り切れるようにしておく, とか). なのでここら辺はフォーマット(preprocessはここに配置)みたいなことだけ統一して後は毎回書く感じがいいかもね.
        - train_dnntts.py: 前回同様. @ stage 5, 6

- vc_tts_template
    - dsp.py: degital signale processorの意味. f0抽出など, 良さげな信号処理関数盛沢山. 汎用的.
    - wavenet
        - conv.py: autoregressive計算用の, buffer機能付きconv1dの実装. 普通にほかでも利用できるし使っていきたい.
            - 簡単に言えば, inference時は確かに同じ部分を計算しまくるので, 少しでも速くするために入力を覚えておく感じ.
            - weight_normはこれにはついていないことに注意.
            - forwardは, nn.Conv1dを引き継いでいるので適用可能.
        - upsample.py: upsampleしたいときに流用出来そう.　一応一般的.
            - module.pyのconvを利用している.
            - ↑このconvはconv.pyに置いたほうがよくないか?
        - modules.py: wavenetのlayerなのでほぼ使いまわせない.
            - mainの繰り返すlayerのとこだけ.
        - wavenet.py: そのまま.