# train_memo
今まで同様に, trainの詳細をここに記載しておく.

- spk
    - jsut
        - 先生にアライメントも二回目の修正をしてもらったきれいなやつ.
    - Long_dialogue
        - 先生にもらったもので, soxで22050にしたもの.
    
    - JSUT_accent
        - tacotronの実装にある, フルコンテキストラベルのアクセント情報を利用したデータセット

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