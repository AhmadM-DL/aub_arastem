#!/bin/bash
#SBATCH --job-name=arastem_eval 

#SBATCH --partition=normal                               

#SBATCH --mem=24000                      
#SBATCH --time=2-00:00:00

#SBATCH --output=output_%j.log         
#SBATCH --error=error_%j.log

module load python/3
source 

export HF_HOME=""
export HF_TOKEN=""

MODEL=""
python download_model.py --model $MODEL
