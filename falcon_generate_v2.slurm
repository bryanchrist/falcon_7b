#!/bin/bash
#SBATCH -A sds-phd-2022
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1 # replace X with the number of GPUs per node
#SBATCH -C gpupod

##SBATCH --hint=nomultithread #8GB per reserved CPU core if hyperthreading deactivated and 1/2 that when active
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:a100_80gb:1
##SBATCH -c 1
#SBATCH -t 0-01:00:00
#SBATCH -J falcon_generate_v2_run1
#SBATCH -o generate_v2-%A.out
#SBATCH -e generate_v2-%A.err
##SBATCH -x udc-an28-[1,7],udc-an34-[1,7,13,19]#,udc-an36-[1,13,19]#,udc-an37-[1,7,13,19]#,udc-ba25-2[3,7,8],udc-ba26-2[3-6],udc-ba27-2[3-4]

module purge

#module load singularity pytorch/1.8.1  # 2
#singularity run --nv $CONTAINERDIR/pytorch-1.8.1.sif finetune/adapter.py # 3

#module load singularity
#singularity run --nv pytorch_23.03-py3.sif python finetune/adapter.py 

module load anaconda
module load cuda/11.4.2
source activate falcon_40B
pip install -U -r requirements.txt
conda install -y cudatoolkit
export LD_LIBRARY_PATH='/opt/conda/lib/' 
python generate.py