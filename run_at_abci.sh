# 例: qsub_Ag1 -l h_rt='50:00:00' -o ~/logs/LINE_wContext_sr22050_LINE_wContext_4_202110230025.log run_at_abci.sh
# 例: qsub_Ag1 -l h_rt='50:00:00' -o ~/logs/LINE_3_sr22050_LINE_6_hifigan_202110221830.log run_at_abci.sh
# 例: qsub_Ag1 -l h_rt='50:00:00' -o ~/logs/LINE_3_sr22050_LINE_12_202110271157.log run_at_abci.sh
# 例: qrsh -g $ABCI_GROUP -l rt_AG.small=1 -l h_rt=10:00:00

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/vc_tts_template/bin/activate

# 具体的処理
# cd /home/acd13708uu/gcb50354/yuto_nishimura/vc_tts_template/recipes/fastspeech2wContexts  # node V
# export PYTHONPATH="/home/acd13708uu/gcb50354/yuto_nishimura/vc_tts_template:$PYTHONPATH"  # node V
# cd /groups/4/gcb50354/migrated_from_SFA_GPFS/yuto_nishimura/vc_tts_template/recipes/fastspeech2wContexts  # node A
cd /groups/4/gcb50354/migrated_from_SFA_GPFS/yuto_nishimura/vc_tts_template/recipes/fastspeech2  # node A
export PYTHONPATH="/groups/4/gcb50354/migrated_from_SFA_GPFS/yuto_nishimura/vc_tts_template:$PYTHONPATH"  # node A

# ./run.sh --stage 3 --stop-stage 3 --local_dir ${SGE_LOCALDIR}/
./run.sh --stage 4 --stop-stage 4 --local_dir ${SGE_LOCALDIR}/
# ./run.sh --stage 5 --stop-stage 5 --local_dir ${SGE_LOCALDIR}/
deactivate
