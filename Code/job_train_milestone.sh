#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=/directory/to/your/training/output/info.out
#SBATCH --error=/directory/to/your/training/output/info.err
#SBATCH --cpus-per-task=16
##SBATCH --mem-per-cpu=32768
#SBATCH --mem=0
#SBATCH --partition=volta-x86
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --qos=long
#SBATCH --time=48:00:00
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID" 
echo "=========================================================="
##cat /etc/redhat-release
cd /directory/to/your/training/
#source /vast/home/hwang/.bashrc
conda activate env_invnet
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6000
echo "Master Node: $MASTER"
echo "Slave Node(s): $SLAVES"

srun python -u train_mlreal_milestone.py \
	-o models -s save_name -n directory_name \
	-m Passive_heat_map_3D_intz \
	--up_mode nearest -ds illinois_heat_map_3D \
	-b 16 -eb 100 -nb 10 -j 1 --lr 1e-4 --tensorboard \
	-g1v 0 -g2v 1 --k 1 \
	--sync-bn --dist-url tcp://$MASTER:$MASTERPORT --world-size 4 \
	-t /directory/to/your/training/train_notch_filter_avg_mask_env_syn_normal_random_intz_heat_map.txt \
	-v /directory/to/your/training/train_notch_filter_avg_mask_env_fd_random_intz_heat_map.txt \
	--resume /directory/to/your/training/models/directory_name/save_name/checkpoint.pth
## -ploss -p1s 1 -p2s 1 -fl 3 \
######clx-volta volta-x86   #ViT_Decoder_Resize FCN4_Aqaba_Resize
