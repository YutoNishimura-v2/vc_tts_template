#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
COMMON_ROOT=../common
. $COMMON_ROOT/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

stage=0
stop_stage=0

. $COMMON_ROOT/parse_options.sh || exit 1;

dumpdir=dump
dump_org_dir=$dumpdir/${spk}_sr${sample_rate}/org
dump_norm_dir=$dumpdir/${spk}_sr${sample_rate}/norm

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data download"
#     mkdir -p downloads
#     if [ ! -d downloads/jsut_ver1 ]; then
#         cd downloads
#         curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
#         unzip -o jsut_ver1.1.zip
#         cd -
#     fi
#     if [ ! -d downloads/jsut-lab ]; then
#         cd downloads
#         curl -LO https://github.com/sarulab-speech/jsut-label/archive/v0.0.2.zip
#         unzip -o v0.0.2.zip
#         ln -s jsut-label-0.0.2 jsut-label
#         cd -
#     fi
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "train/dev/eval split"
    mkdir -p data
    find $lab_root -name "*.TextGrid" -exec basename {} .TextGrid \; | shuf > data/utt_list.txt
    head -n 6242 data/utt_list.txt > data/train.list
    tail -300 data/utt_list.txt > data/deveval.list
    head -n 200 data/deveval.list > data/dev.list
    tail -n 100 data/deveval.list > data/eval.list
    rm -f data/deveval.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for fastspeech2"
    for s in ${datasets[@]}; do
        xrun python preprocess.py data/$s.list $wav_root $lab_root \
            $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate --filter_length $filter_length \
            --hop_length $hop_length --win_length $win_length \
            --n_mel_channels $n_mel_channels --mel_fmin $mel_fmin --mel_fmax $mel_fmax \
            --multi_speaker $multi_speaker --pitch_phoneme_averaging $pitch_phoneme_averaging \
            --energy_phoneme_averaging $energy_phoneme_averaging
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature normalization"
    for typ in "fastspeech2"; do
       for inout in "out"; do
            for feat in "mel" "pitch" "energy"; do
                xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
                    $dump_org_dir/$train_set/${inout}_${typ}/${feat} \
                    $dump_org_dir/${inout}_${typ}_${feat}_scaler.joblib
            done
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "fastspeech2"; do
            for inout in "out" "in"; do
                if [ $inout == "in" ]; then
                    cp -r $dump_org_dir/$s/${inout}_${typ} $dump_norm_dir/$s/
                    continue
                fi
                for feat in "mel" "pitch" "energy" "duration"; do
                    if [ $feat == "duration" ]; then
                        cp -r $dump_org_dir/$s/${inout}_${typ}/${feat} $dump_norm_dir/$s/${inout}_${typ}
                        continue
                    fi
                    xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
                        $dump_org_dir/${inout}_${typ}_${feat}_scaler.joblib \
                        $dump_org_dir/$s/${inout}_${typ}/${feat} \
                        $dump_norm_dir/$s/${inout}_${typ}/${feat} --n_jobs $n_jobs
                done
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: finetuning hifigan"
    xrun python train_hifigan.py model=$vocoder_model tqdm=$tqdm \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$wav_root \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$wav_root \
        data.batch_size=$hifigan_data_batch_size \
        data.sampling_rate=$sample_rate \
        data.n_fft=$filter_length \
        data.num_mels=$n_mel_channels \
        data.hop_size=$hop_length \
        data.win_size=$win_length \
        data.fmin=$mel_fmin \
        data.fmax=$mel_fmax \
        train.out_dir=$expdir/${vocoder_model} \
        train.log_dir=tensorboard/${expname}_${vocoder_model} \
        train.max_train_steps=$hifigan_train_max_train_steps
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training fastspeech2"
    xrun python train_fastspeech2.py model=$acoustic_model tqdm=$tqdm \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_fastspeech2/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_fastspeech2/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_fastspeech2/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_fastspeech2/ \
        data.batch_size=$fastspeech2_data_batch_size \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.max_train_steps=$fastspeech2_train_max_train_steps \
        train.sampling_rate=$sample_rate \
        train.vocoder_name=$vocoder_model \
        train.criterion.pitch_feature_level=$pitch_phoneme_averaging\
        train.criterion.energy_feature_level=$energy_phoneme_averaging\
        model.netG.pitch_feature_level=$pitch_phoneme_averaging \
        model.netG.energy_feature_level=$energy_phoneme_averaging \
        model.netG.n_mel_channel=$n_mel_channels \
        model.netG.multi_speaker=$multi_speaker
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 6: Synthesis waveforms by hifigan"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=${lab_root} \
            out_dir=$expdir/synthesis_${acoustic_model}_${wavenet_model}/$s \
            sample_rate=$sample_rate \
            acoustic.checkpoint=$expdir/${acoustic_model}/$acoustic_eval_checkpoint \
            acoustic.out_scaler_path=$dump_norm_dir/out_tacotron_scaler.joblib \
            acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
            wavenet.checkpoint=$expdir/${wavenet_model}/$wavenet_eval_checkpoint \
            wavenet.model_yaml=$expdir/${wavenet_model}/model.yaml \
            use_wavenet=true reverse=$reverse num_eval_utts=$num_eval_utts
    done
fi

if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ]; then
    echo "Create tar.gz to share experiments"
    rm -rf tmp/exp
    mkdir -p tmp/exp/$expname
    for model in $acoustic_model $wavenet_model; do
        rsync -avr $expdir/$model tmp/exp/$expname/ --exclude "epoch*.pth"
    done
    rsync -avr $expdir/synthesis_${acoustic_model}_griffin_lim tmp/exp/$expname/ --exclude "epoch*.pth"
    rsync -avr $expdir/synthesis_${acoustic_model}_${wavenet_model} tmp/exp/$expname/ --exclude "epoch*.pth"
    cd tmp
    tar czvf tacotron_exp.tar.gz exp/
    mv tacotron_exp.tar.gz ..
    cd -
    rm -rf tmp
    echo "Please check tacotron_exp.tar.gz"
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${acoustic_model}_${wavenet_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
mu: ${mu}
acoustic_model: ${acoustic_model}
wavenet_model: ${wavenet_model}
EOL

    # Stats
    python $COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/out_tacotron_scaler.joblib $dst_dir

    # Acoustic model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
        $dst_dir/acoustic_model.pth
    cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

    # WaveNet
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${wavenet_model}/$wavenet_eval_checkpoint \
        $dst_dir/wavenet_model.pth
    cp $expdir/${wavenet_model}/model.yaml $dst_dir/wavenet_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
fi