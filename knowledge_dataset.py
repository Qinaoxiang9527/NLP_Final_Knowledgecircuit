# ... existing code ...

def load_dataset(paths):
    """Load relations from json files in a folder.

    Accepts one or more directories or files. If a file, should be JSON format, and will
    be read as one relation. If a directory, will recursively search for all JSON files.
    """
    # Load all relation files
    files = []
    for path in paths:
        files.append(path)
    logger.debug(f"found {len(files)} relation files total, loading...")
    relation_dicts = [load_relation_dict(file) for file in files]
    # Mark all disambiguating relations
    domain_range_pairs: dict[tuple[str, str], int] = {}
    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        cur = domain_range_pairs.get((d, r), 0)
        domain_range_pairs[(d, r)] = cur + 1

    for relation_dict in relation_dicts:
        d, r = (
            relation_dict["properties"]["domain_name"],
            relation_dict["properties"]["range_name"],
        )
        relation_dict["properties"]["disambiguating"] = domain_range_pairs[(d, r)] > 1

    # Create Relation objects
    relations = [Relation.from_dict(relation_dict) for relation_dict in relation_dicts]

    return relations

# 添加加载本体知识数据集的函数
def load_ontology_datasets():
    """加载转换后的本体知识数据集"""
    from pathlib import Path
    import os
    
    datasets = []
    base_dir = Path(__file__).parent.parent.parent / "data"
    
    # 加载Ontology文件夹中的转换后数据
    ontology_dir = base_dir / "Ontology"
    ontology_files = list(ontology_dir.glob("*_converted.json"))
    
    # 加载Memorizing_converted文件夹中的转换后数据
    memorizing_dir = base_dir / "Memorizing_converted"
    if memorizing_dir.exists():
        ontology_files.extend(memorizing_dir.glob("*.json"))
    
    # 如果找不到转换后的文件，尝试运行转换脚本
    if not ontology_files:
        try:
            from ontology_adapter import convert_class_json, convert_property_json, convert_memorizing_jsonl
            convert_class_json()
            convert_property_json()
            convert_memorizing_jsonl()
            
            # 重新查找转换后的文件
            ontology_files = list(ontology_dir.glob("*_converted.json"))
            if memorizing_dir.exists():
                ontology_files.extend(memorizing_dir.glob("*.json"))
        except ImportError:
            logger.warning("无法导入ontology_adapter模块，请确保已创建该模块")
    
    # 加载每个文件
    for file_path in ontology_files:
        try:
            with open(file_path, 'r') as f:
                relation_dict = json.load(f)
                relation = Relation.from_dict(relation_dict)
                datasets.append(relation)
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {e}")
    
    return datasets

# ... existing code ...