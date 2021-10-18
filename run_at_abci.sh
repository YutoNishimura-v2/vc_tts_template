source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.1/11.1.1 cudnn/8.0/8.0.5
source ~/venv/vc_tts_template/bin/activate

# 具体的処理
# cd /home/acd13708uu/gcb50354/yuto_nishimura/vc_tts_template/recipes/fastspeech2wContexts  # node V
# export PYTHONPATH="/home/acd13708uu/gcb50354/yuto_nishimura/vc_tts_template:$PYTHONPATH"  # node V
cd /groups/4/gcb50354/migrated_from_SFA_GPFS/yuto_nishimura/vc_tts_template/recipes/fastspeech2wContexts  # node A
export PYTHONPATH="/groups/4/gcb50354/migrated_from_SFA_GPFS/yuto_nishimura/vc_tts_template:$PYTHONPATH"  # node A

# ./run.sh --stage 4 --stop-stage 4
./run.sh --stage 4 --stop-stage 4 --local_dir ${SGE_LOCALDIR}/
deactivate
