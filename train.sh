#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32GB

mamba init bash
source ~/.bashrc

conda activate ALBEF

export TOKENIZERS_PARALLELISM=false

python CLOVE.py \
--config ./configs/VQA.yaml \
--output_dir output/vqa \
--checkpoint albef_vqav2_lavis.pt \
--order abcdef \
--dual_cluster_prompt True \
--data_root /project/rostamim_919/caiyulia/data/CLOVE/