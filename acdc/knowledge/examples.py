import logging
import torch
from pathlib import Path

from acdc.knowledge.ontology_dataset import OntologyDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 设置数据目录为绝对路径
    data_dir = Path("/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data")
    
    # 创建本体数据集加载器
    ontology_dataset = OntologyDataset(data_dir)
    
    # 加载所有关系
    relation_dataset = ontology_dataset.load_all_relations()
    
    # 打印加载的关系信息
    logger.info(f"加载了{len(relation_dataset)}个关系")
    
    # 打印每个关系的详细信息
    for i, relation in enumerate(relation_dataset):
        logger.info(f"关系 {i+1}: {relation.name}")
        logger.info(f"  样本数量: {len(relation.samples)}")
        logger.info(f"  领域大小: {len(relation.domain)}")
        logger.info(f"  值域大小: {len(relation.range)}")
        logger.info(f"  函数类型: {relation.properties.fn_type}")
        
        # 打印前5个样本
        for j, sample in enumerate(relation.samples[:5]):
            logger.info(f"  样本 {j+1}: {sample}")
            
        logger.info("---")

if __name__ == "__main__":
    main()