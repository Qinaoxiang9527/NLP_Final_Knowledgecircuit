#!/bin/bash
#SBATCH --job-name=knowledgecircuits
#SBATCH --partition=lab-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output result.out         ## filename of the output      
#SBATCH --cpus-per-task=16           ## Number of CPU cores allocated to each process
#SBATCH --gres=gpu:1                ## Use 1 gpu


MODEL_PATH=/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/acdc/models/gpt2-medium
KT=factual
KNOWLEDGE=country_capital_city
NUM_EXAMPLES=1
MODEL_NAME=gpt2-medium

python main.py --task=knowledge \
--zero-ablation \
--threshold=0.01 \
--device=cuda:0 \
--metric=match_nll \
--indices-mode=reverse \
--first-cache-cpu=False \
--second-cache-cpu=False \
--max-num-epochs=100000 \
--specific-knowledge=$KNOWLEDGE \
--num-examples=$NUM_EXAMPLES \
--relation-reverse=False \
--knowledge-type=$KT \
--model-name=$MODEL_NAME \
--model-path=$MODEL_PATH
