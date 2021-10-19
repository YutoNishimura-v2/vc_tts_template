# train_memo
今まで同様に, trainの詳細をここに記載しておく.

- spk
    - jsut
        - 先生にアライメントも二回目の修正をしてもらったきれいなやつ.
    - Long_dialogue
        - 先生にもらったもので, soxで22050にしたもの.
    - JSUT_accent
        - tacotronの実装にある, フルコンテキストラベルのアクセント情報を利用したデータセット
    - LINE
        - corpus評価用のwoITAKO.
    - LINE_2
        - corpus評価用のwoITAKO.
        - statsの計算方法を変えたので, 一応作り直し.
    - LINE_3
        - corpus評価用のwoITAKO.
        - accent info付き.
    - LINE_4
        - corpus評価用のwoITAKO_before_emotion.
        - accent info付き.
        - emotionを変えたのでdataをコピーしなおしていることに注意
    
    - JSUT_NICT_LINE
        - VCのpretrain用の大規模TTS. textgrid.

- exp
    - jsut_1
        - Long_dialogueのための, pretrain.
    - Long_dialogue_1
        - emotionを実装してからの初実行. 比較対象がないのが残念.
    - Long_dialogue_2
        - emotionを外して. Long_dialogue_1と全く同じtrain_set.
        - さらに, hifiganも完全に使いまわし.
    - Long_dialogue_3
        - pretrainだけなしにして, それ以外は2と同じ.
        - lossはほぼ同じだけど, 音質は若干悪い感じ.
    - Long_dialogue_4
        - spk: Long_dialogue
        - pretrain: None
        - wGMM初稼働. ミスって上書きしてしまった...。
    - Long_dialogue_5
        - spk: Long_dialogue
        - pretrain: None
        - wGMM実質初稼働. ミスって上書きしてしまった...。
    - Long_dialogue_6
        - spk: Long_dialogue
        - pretrain: None
        - wGMM+global prosody.
    - Long_dialogue_7
        - spk: Long_dialogue
        - pretrain: None
        - optunaでハイパラ探索を行う
    - JSUT_accent_1(@myPC)
        - spk: JSUT_accent
        - pretrain: None
        - アクセント付きFastSpeech2初実行
    - Long_dialogue_8
        - spk: Long_dialogue
        - pretrain: None
        - optunaでハイパラ探索を行う. 別パラメタ。アドバイスを受けて大分変数を減らした.
    - Long_dialogue_9
        - spk: Long_dialogue
        - pretrain: None
        - optunaでハイパラ探索を行う. 別パラメタ。
            - prunerを変えてnum_gaussianをカテゴリ―にした.
    
    - ここで、VCの方にシフト. lr_scheduler周りで2つほどミスを見つけたりの修正が出来たので、ましになるかも.
    - wGMMも他のも, アクセントを前提にしてみる.
    - LINE_1
        - spk: LINE
        - pretrain: None
        - ふっつーのfastspeech2.
    - LINE_2
        - spk: LINE_2
        - pretrain: None
        - 29min/50epoch
        - batch_size: 8
        - ふっつーのfastspeech2. LINE_1でstatsが正規化前のものになっているのに気づかず使っていた. そこを修正して実行しなおしたもの.
    - LINE_3
        - spk: LINE_3
        - pretrain: None
        - +accent info
        - batch_size: 8
        - loss上は微改善. 音声も劇的ではないが、改善.
    - LINE_4
        - spk: LINE_3
        - pretrain: None
        - +accent info + emotion
        - batch_size: 8
    - LINE_5
        - spk: LINE_4
        - pretrain: None
        - +accent info + before_emotion
        - batch_size: 8
    
    - 以下は趣味のVC用.
    - JSUT_NICT_LINE_1
        - spk: JSUT_NICT_LINE
        - pretrain用.

## 主要な実験

## 知見