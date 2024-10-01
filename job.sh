#!/bin/bash
#SBATCH --job-name=arastem_eval 

#SBATCH --partition=gpu                               
#SBATCH --gres=gpu:v100d32q:2                  

#SBATCH --mem=24000                      
#SBATCH --time=06:00:00

#SBATCH --output=output_%j.log         
#SBATCH --error=error_%j.log

module load python/3
source /home/am252/scratch/myenv/bin/activate

MODEL=""
DATA="/home/am252/scratch/aub_arastem/data/arastem.json"
MAX_INPUT_TOKENS=2048
HF_CACHE="/home/am252/scratch/hf_models/"
python run.py --model $MODEL --data $DATA -s $MAX_INPUT_TOKENS -hf $HF_CACHE -v

