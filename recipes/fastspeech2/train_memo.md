# train_memo
今まで同様に, trainの詳細をここに記載しておく.

- spk
    - jsut
        - 先生にアライメントも二回目の修正をしてもらったきれいなやつ.
    - Long_dialogue
        - 先生にもらったもので, soxで22050にしたもの.

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
