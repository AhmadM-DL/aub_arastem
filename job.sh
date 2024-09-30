#!/bin/bash
#SBATCH --job-name=arastem_eval 

#SBATCH --partition=gpu
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                                
#SBATCH --gres=gpu:v100d32q:4                   

#SBATCH --mem=32000                      
#SBATCH --time=06:00:00                
#SBATCH --output=output_%j.log         
#SBATCH --error=error_%j.log

module load python/3
source /home/am252/scratch/myenv/bin/activate

MODEL = "inceptionai/jais-30b-v1"
DATA = "/home/am252/scratch/arastem/data/arastem.json"
MAX_INPUT_TOKENS = 2048
python run.py --model $MODEL --data $DATA -s $MAX_INPUT_TOKENS -v

