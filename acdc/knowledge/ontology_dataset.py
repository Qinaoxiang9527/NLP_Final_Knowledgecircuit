import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .knowledge_dataset import Relation, RelationDataset, RelationSample, RelationProperties

logger = logging.getLogger(__name__)

class OntologyDataset(RelationDataset):
    """用于处理本体知识数据集的类，继承自RelationDataset"""
    
    def __init__(self, relations=None):
        """
        初始化本体知识数据集
        
        Args:
            relations: 关系列表，默认为空列表
        """
        super().__init__(relations or [])
        self.ontology_types = {
            "subClassOf", "subPropertyOf", "type", "domain", "range"
        }
    
    @classmethod
    def from_files(cls, files: List[str]) -> 'OntologyDataset':
        """
        从文件列表加载本体知识数据集
        
        Args:
            files: 文件路径列表
            
        Returns:
            OntologyDataset: 加载的本体知识数据集
        """
        relations = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    relation_dict = json.load(f)
                    relation = Relation.from_dict(relation_dict)
                    relations.append(relation)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
        
        return cls(relations)
    
    @classmethod
    def load_ontology_datasets(cls) -> 'OntologyDataset':
        """
        加载转换后的本体知识数据集
        
        Returns:
            OntologyDataset: 加载的本体知识数据集
        """
        from pathlib import Path
        
        datasets = []
        base_dir = Path(__file__).parent.parent.parent / "data"
        
        # 加载Ontology文件夹中的转换后数据
        ontology_dir = base_dir / "ontology"
        ontology_files = list(ontology_dir.glob("*_converted.json"))
        
        # 加载Memorizing_converted文件夹中的转换后数据
        memorizing_dir = base_dir / "Memorizing_converted"
        if memorizing_dir.exists():
            ontology_files.extend(memorizing_dir.glob("*.json"))
        
        # 如果找不到转换后的文件，尝试运行转换脚本
        if not ontology_files:
            try:
                # 尝试导入转换模块并运行转换函数
                import sys
                sys.path.append(str(base_dir.parent))
                from convert_ontology import convert_class_json, convert_property_json, convert_memorizing_jsonl
                
                convert_class_json()
                convert_property_json()
                convert_memorizing_jsonl()
                
                # 重新查找转换后的文件
                ontology_files = list(ontology_dir.glob("*_converted.json"))
                if memorizing_dir.exists():
                    ontology_files.extend(memorizing_dir.glob("*.json"))
            except ImportError:
                logger.warning("无法导入转换模块，请确保已创建convert_ontology.py文件")
            except Exception as e:
                logger.error(f"运行转换脚本时出错: {e}")
        
        # 加载每个文件
        for file_path in ontology_files:
            try:
                with open(file_path, 'r') as f:
                    relation_dict = json.load(f)
                    relation = Relation.from_dict(relation_dict)
                    datasets.append(relation)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
        
        return cls(datasets)
    
    def get_relation_by_type(self, relation_type: str) -> List[Relation]:
        """
        根据关系类型获取关系列表
        
        Args:
            relation_type: 关系类型，如"subClassOf"、"type"等
            
        Returns:
            List[Relation]: 指定类型的关系列表
        """
        return [r for r in self if r.properties.relation_type == relation_type]
    
    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """
        获取类层次结构
        
        Returns:
            Dict[str, List[str]]: 类及其子类的映射
        """
        hierarchy = {}
        for relation in self.get_relation_by_type("subClassOf"):
            for sample in relation.samples:
                parent = sample.object
                child = sample.subject
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(child)
        return hierarchy
    
    def get_property_hierarchy(self) -> Dict[str, List[str]]:
        """
        获取属性层次结构
        
        Returns:
            Dict[str, List[str]]: 属性及其子属性的映射
        """
        hierarchy = {}
        for relation in self.get_relation_by_type("subPropertyOf"):
            for sample in relation.samples:
                parent = sample.object
                child = sample.subject
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(child)
        return hierarchy
    
    def get_instances_of_class(self, class_name: str) -> List[str]:
        """
        获取指定类的所有实例
        
        Args:
            class_name: 类名
            
        Returns:
            List[str]: 实例列表
        """
        instances = []
        for relation in self.get_relation_by_type("type"):
            for sample in relation.samples:
                if sample.object == class_name:
                    instances.append(sample.subject)
        return instances
    
    def get_domain_of_property(self, property_name: str) -> List[str]:
        """
        获取属性的定义域
        
        Args:
            property_name: 属性名
            
        Returns:
            List[str]: 定义域类列表
        """
        domains = []
        for relation in self.get_relation_by_type("domain"):
            for sample in relation.samples:
                if sample.subject == property_name:
                    domains.append(sample.object)
        return domains
    
    def get_range_of_property(self, property_name: str) -> List[str]:
        """
        获取属性的值域
        
        Args:
            property_name: 属性名
            
        Returns:
            List[str]: 值域类列表
        """
        ranges = []
        for relation in self.get_relation_by_type("range"):
            for sample in relation.samples:
                if sample.subject == property_name:
                    ranges.append(sample.object)
        return ranges
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将本体知识数据集转换为字典
        
        Returns:
            Dict[str, Any]: 数据集字典
        """
        return {
            "class_hierarchy": self.get_class_hierarchy(),
            "property_hierarchy": self.get_property_hierarchy(),
            "relations": [r.to_dict() for r in self]
        }
    
    def save_to_file(self, file_path: str) -> None:
        """
        将本体知识数据集保存到文件
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ontologyDataset':
        """
        从字典创建本体知识数据集
        
        Args:
            data: 数据集字典
            
        Returns:
            OntologyDataset: 创建的本体知识数据集
        """
        relations = [Relation.from_dict(r) for r in data.get("relations", [])]
        return cls(relations)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ontologyDataset':
        """
        从文件加载本体知识数据集
        
        Args:
            file_path: 文件路径
            
        Returns:
            OntologyDataset: 加载的本体知识数据集
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    # 测试代码
    ontology_dataset = OntologyDataset.load_ontology_datasets()
    print(f"加载了 {len(ontology_dataset)} 个本体知识关系")
    
    # 打印每个关系的样本数量
    for relation in ontology_dataset:
        print(f"关系: {relation.name}, 样本数量: {len(relation.samples)}")
    
    # 打印类层次结构
    class_hierarchy = ontology_dataset.get_class_hierarchy()
    print(f"类层次结构: {len(class_hierarchy)} 个父类")
    
    # 打印属性层次结构
    property_hierarchy = ontology_dataset.get_property_hierarchy()
    print(f"属性层次结构: {len(property_hierarchy)} 个父属性")