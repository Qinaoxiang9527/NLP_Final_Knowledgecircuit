import os
import json
from acdc.knowledge.knowledge_dataset import RelationSample, RelationProperties, Relation, RelationDataset

def load_ontology_datasets():
    """加载转换后的本体知识数据集"""
    datasets = []
    
    # 加载Ontology文件夹中的转换后数据
    ontology_files = [
        "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/class_converted.json",
        "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/property_converted.json"
    ]
    
    # 加载Memorizing_converted文件夹中的转换后数据
    memorizing_dir = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/Memorizing_converted"
    if os.path.exists(memorizing_dir):
        for filename in os.listdir(memorizing_dir):
            if filename.endswith(".json"):
                ontology_files.append(os.path.join(memorizing_dir, filename))
    
    # 加载每个文件并创建Relation对象
    for file_path in ontology_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # 创建RelationProperties对象
                properties = RelationProperties(
                    relation_type=data["properties"]["relation_type"],
                    domain_name=data["properties"]["domain_name"],
                    range_name=data["properties"]["range_name"],
                    symmetric=data["properties"]["symmetric"],
                    fn_type=data["properties"]["fn_type"],
                    disambiguating=data["properties"]["disambiguating"]
                )
                
                # 创建RelationSample对象列表
                samples = [
                    RelationSample(subject=sample["subject"], object=sample["object"])
                    for sample in data["samples"]
                ]
                
                # 创建Relation对象
                relation = Relation(
                    name=data["name"],
                    prompt_templates=data["prompt_templates"],
                    prompt_templates_zs=data["prompt_templates_zs"],
                    samples=samples,
                    properties=properties
                )
                
                datasets.append(relation)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    return RelationDataset(datasets)

# 使用示例
if __name__ == "__main__":
    ontology_dataset = load_ontology_datasets()
    print(f"加载了 {len(ontology_dataset)} 个本体知识关系")
    
    # 打印每个关系的样本数量
    for relation in ontology_dataset:
        print(f"关系: {relation.name}, 样本数量: {len(relation.samples)}")