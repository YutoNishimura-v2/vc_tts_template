# train_memo
今まで同様に, trainの詳細をここに記載しておく.

- spk
    - jsut_jsss
        - 最初の.
        - pitchの実装をミスってしまい, 0になっているやつ
    
    - jsut_jsss_1
        - ↑を修正したもの

    - N2C
        - N2C初回.
    
    - jsut_jsss_jvs
        - 上に書いてあるコーパスすべてのVC. multi.
        - pretrain用.
    
    - N2C_2
        - sent_durationを実装して再実行.

- tag
    - jsut_jsss_1
        - spk: jsut_jsss
    
    - jsut_jsss_2
        - spk: jsut_jsss_1
        - pitchを修正して初挑戦
    
    - N2C_1
        - spk: N2C
        - pretrain: jsut_jsss_2
    
    - jsut_jsss_jvs_1
        - spk: jsut_jsss_jvs
        - pretrain用.

    - N2C_2
        - spk: N2C
        - pretrian: jsut_jsss_jvs_1

    - N2C_3
        - spk: N2C
        - pretrain: なし
        - VariancePridicterでreduction factorの実装ミスっていた説があるので実行しなおし.
    
    - N2C_4
        - spk: N2C
        - pretrain: なし
        - pitchのAR化を実装した.
    
    - N2C_5
        - spk: N2C_2
        - pretrain: なし
        - pitchのAR化を実装した.