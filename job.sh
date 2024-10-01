#!/bin/bash
#SBATCH --job-name=arastem_eval 

#SBATCH --partition=gpu                               
#SBATCH --gres=gpu:v100d32q:2                  

#SBATCH --mem=24000                      
#SBATCH --time=06:00:00

#SBATCH --output=output_%j.log         
#SBATCH --error=error_%j.log

module load python/3
source 

export HF_HOME=""
export HF_TOKEN=""

MODEL=""
DATA=""
MAX_INPUT_TOKENS=
python run.py --model $MODEL --data $DATA -s $MAX_INPUT_TOKENS -v

