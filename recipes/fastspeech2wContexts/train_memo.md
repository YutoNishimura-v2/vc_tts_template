# train_memo

- spk
    - LINE_wContext
        - out_LINE_woITAKOから作ったもの.
        - dialogue_infoにTurn-00としてsituation情報を追加
            - これがないと, 先頭のhistoryが0になって全部0でエラーとかが起きる.
            - あとは今になって思えばこの情報はかなり重要そう
- exp
    - LINE_wContext_1
        - spk: LINE_wContext
        - pretrain: None
        - +accent
        - 初実行. wEmotionは次かな?
    - LINE_wContext_2
        - spk: LINE_wContext
        - pretrain: None
        - +accent+Emotion
        - 初実行. wEmotionは次かな?