#!/bin/bash
#SBATCH --job-name=ontologycircuits
#SBATCH --partition=lab-4090
 #SBATCH --nodes=1
#SBATCH --ntasks=1                         ## 使用一个任务
#SBATCH --output ontology_result.out        ## 输出文件名      
#SBATCH --cpus-per-task=16                  ## 分配给每个进程的CPU核心数
#SBATCH --gres=gpu:1                        ## 使用一个GPU

# 设置模型和数据参数
MODEL_PATH=/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/acdc/models/gpt2-medium
MODEL_NAME=gpt2-medium
DATA_PATH=/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data  # 数据根目录
RELATION_NAME="ontology_class_hierarchy"  # 可以是 ontology_class_hierarchy 或 ontology_property_hierarchy
NUM_EXAMPLES=10
OUTPUT_DIR="acdc/ontology_results"

# 确保输出目录存在
mkdir -p ${OUTPUT_DIR}/${MODEL_NAME}/${RELATION_NAME}

# 运行本体知识实验
python ontology_main.py \
--threshold=0.2 \
--device=cuda \
--model-name=${MODEL_NAME} \
--model-path=${MODEL_PATH} \
--data-path=${DATA_PATH} \
--relation-name=${RELATION_NAME} \
--num-examples=${NUM_EXAMPLES} \
--metric=match_nll \
--indices-mode=normal \
--zero-ablation \
--first-cache-cpu=True \
--second-cache-cpu=True \
--max-num-epochs=5000 \
--visualize \
--output-dir=${OUTPUT_DIR} \
--colorscheme=Pastel2