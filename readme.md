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