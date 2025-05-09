import json
import os

# 转换Ontology/class.json
def convert_class_json():
    input_path = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/class.json"
    output_path = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/class_converted.json"
    
    with open(input_path, 'r') as f:
        class_data = json.load(f)
    
    # 创建符合原有数据集格式的结构
    converted_data = {
        "name": "ontology_class_hierarchy",
        "prompt_templates": [
            "In ontology, {} is a subclass of",
            "{} is a type of",
            "{} belongs to the class of"
        ],
        "prompt_templates_zs": [
            "In ontology, {} is a subclass of",
            "{} is a type of",
            "{} belongs to the class of"
        ],
        "samples": [],
        "properties": {
            "relation_type": "subClassOf",
            "domain_name": "Class",
            "range_name": "Class",
            "symmetric": False,
            "fn_type": "MANY_TO_ONE",
            "disambiguating": False
        }
    }
    
    # 将class.json中的数据转换为samples
    for item in class_data:
        # 检查是否有rdfs:subClassOf字段
        if "rdfs:subClassOf" in item:
            subject = item.get("rdfs:label", "") or item.get("id", "")
            for parent in item["rdfs:subClassOf"]:
                # 检查parent是字典还是字符串
                if isinstance(parent, dict):
                    parent_label = parent.get("rdfs:label", "") or parent.get("id", "")
                else:
                    # 如果parent是字符串，直接使用
                    parent_label = parent
                
                if subject and parent_label:
                    converted_data["samples"].append({
                        "subject": subject,
                        "object": parent_label
                    })
                    print(f"添加类层次关系: {subject} -> {parent_label}")
        
        # 检查是否有example字段
        if "example" in item and isinstance(item["example"], list):
            class_name = item.get("rdfs:label", "") or item.get("id", "")
            for example in item["example"]:
                if isinstance(example, list) and len(example) == 2:
                    instance, cls = example
                    if cls == class_name:  # 确保这是该类的实例
                        converted_data["samples"].append({
                            "subject": instance,
                            "object": cls
                        })
                        print(f"添加类实例: {instance} -> {cls}")
    
    # 保存转换后的数据
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"转换完成：{output_path}，共提取了{len(converted_data['samples'])}个样本")

# 转换Ontology/property.json
def convert_property_json():
    input_path = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/property.json"
    output_path = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/ontology/property_converted.json"
    
    with open(input_path, 'r') as f:
        property_data = json.load(f)
    
    # 创建符合原有数据集格式的结构
    converted_data = {
        "name": "ontology_property_hierarchy",
        "prompt_templates": [
            "In ontology, {} is a subproperty of",
            "{} is a type of property related to",
            "The property {} is a specialization of"
        ],
        "prompt_templates_zs": [
            "In ontology, {} is a subproperty of",
            "{} is a type of property related to",
            "The property {} is a specialization of"
        ],
        "samples": [],
        "properties": {
            "relation_type": "subPropertyOf",
            "domain_name": "Property",
            "range_name": "Property",
            "symmetric": False,
            "fn_type": "MANY_TO_ONE",
            "disambiguating": False
        }
    }
    
    # 将property.json中的数据转换为samples
    for item in property_data:
        # 1. 从rdfs:subPropertyOf提取层次关系
        if "rdfs:subPropertyOf" in item:
            subject = item.get("rdfs:label", "")
            # 检查rdfs:subPropertyOf是字典还是字符串或列表
            if isinstance(item["rdfs:subPropertyOf"], dict):
                for parent_key in item["rdfs:subPropertyOf"].keys():
                    if subject and parent_key:
                        converted_data["samples"].append({
                            "subject": subject,
                            "object": parent_key
                        })
                        print(f"添加层次关系: {subject} -> {parent_key}")
            elif isinstance(item["rdfs:subPropertyOf"], list):
                for parent in item["rdfs:subPropertyOf"]:
                    if isinstance(parent, dict):
                        parent_label = parent.get("rdfs:label", "") or parent.get("id", "")
                    else:
                        parent_label = parent
                    
                    if subject and parent_label:
                        converted_data["samples"].append({
                            "subject": subject,
                            "object": parent_label
                        })
                        print(f"添加层次关系: {subject} -> {parent_label}")
            else:
                # 如果是字符串，直接使用
                parent_label = item["rdfs:subPropertyOf"]
                if subject and parent_label:
                    converted_data["samples"].append({
                        "subject": subject,
                        "object": parent_label
                    })
                    print(f"添加层次关系: {subject} -> {parent_label}")
        
        # 2. 从example提取示例数据
        if "example" in item and isinstance(item["example"], list):
            property_name = item.get("rdfs:label", "")
            for example_pair in item["example"]:
                if len(example_pair) == 2:
                    subject, obj = example_pair
                    # 创建一个新的样本
                    sample = {
                        "subject": subject,
                        "object": obj
                    }
                    # 将样本添加到samples列表中
                    converted_data["samples"].append(sample)
                    print(f"添加示例: {subject} -> {obj}")
    
    # 保存转换后的数据
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"转换完成：{output_path}，共提取了{len(converted_data['samples'])}个样本")

# 转换Memorizing文件夹中的JSONL文件
def convert_memorizing_jsonl():
    # 尝试两种可能的路径
    base_dirs = [
        "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/memorizing",
        "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/Memorizing"
    ]
    
    base_dir = None
    for dir_path in base_dirs:
        if os.path.exists(dir_path):
            base_dir = dir_path
            print(f"找到有效路径: {base_dir}")
            break
    
    if not base_dir:
        print("未找到Memorizing目录，跳过转换")
        return
        
    output_dir = "/mnt/workspace/qinaoxiang/KnowledgeCircuits-main/data/Memorizing_converted"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义要处理的文件及其对应的关系类型
    files_to_process = {
        "subClassOf.jsonl": {
            "name": "memorizing_subClassOf",
            "relation_type": "subClassOf",
            "domain_name": "Class",
            "range_name": "Class",
            "prompt_templates": [
                "In ontology, {} is a subclass of",
                "{} is a type of",
                "{} belongs to the class of"
            ]
        },
        "subPropertyOf.jsonl": {
            "name": "memorizing_subPropertyOf",
            "relation_type": "subPropertyOf",
            "domain_name": "Property",
            "range_name": "Property",
            "prompt_templates": [
                "In ontology, {} is a subproperty of",
                "{} is a type of property related to",
                "The property {} is a specialization of"
            ]
        },
        "type.jsonl": {
            "name": "memorizing_type",
            "relation_type": "type",
            "domain_name": "Instance",
            "range_name": "Class",
            "prompt_templates": [
                "{} is an instance of",
                "{} has type",
                "{} belongs to the class"
            ]
        },
        "domain.jsonl": {
            "name": "memorizing_domain",
            "relation_type": "domain",
            "domain_name": "Property",
            "range_name": "Class",
            "prompt_templates": [
                "The property {} has domain",
                "{} is applicable to instances of",
                "The domain of {} is"
            ]
        },
        "range.jsonl": {
            "name": "memorizing_range",
            "relation_type": "range",
            "domain_name": "Property",
            "range_name": "Class",
            "prompt_templates": [
                "The property {} has range",
                "The values of {} are instances of",
                "The range of {} is"
            ]
        }
    }
    
    for filename, config in files_to_process.items():
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jsonl", ".json"))
        
        print(f"处理文件: {input_path}")
        print(f"文件是否存在: {os.path.exists(input_path)}")
        
        # 创建符合原有数据集格式的结构
        converted_data = {
            "name": config["name"],
            "prompt_templates": config["prompt_templates"],
            "prompt_templates_zs": config["prompt_templates"],
            "samples": [],
            "properties": {
                "relation_type": config["relation_type"],
                "domain_name": config["domain_name"],
                "range_name": config["range_name"],
                "symmetric": False,
                "fn_type": "MANY_TO_ONE",
                "disambiguating": False
            }
        }
        
        # 读取JSONL文件并转换
        sample_count = 0
        try:
            with open(input_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        # 尝试多种可能的字段名称
                        subject = None
                        object_value = None
                        
                        # 尝试不同的字段名称组合
                        if "subject" in data and "object" in data:
                            subject = data["subject"]
                            object_value = data["object"]
                        elif "uuu" in data and "xxx" in data:
                            subject = data["uuu"]
                            object_value = data["xxx"]
                        elif "s" in data and "o" in data:
                            subject = data["s"]
                            object_value = data["o"]
                        
                        # 如果找到了主体和对象，添加到样本中
                        if subject and object_value:
                            # 如果object是列表，取第一个元素
                            if isinstance(object_value, list) and len(object_value) > 0:
                                object_value = object_value[0]
                                
                            converted_data["samples"].append({
                                "subject": subject,
                                "object": object_value
                            })
                            sample_count += 1
                        else:
                            print(f"第{line_num}行缺少主体或对象字段: {line.strip()}")
                    except json.JSONDecodeError as e:
                        print(f"第{line_num}行JSON解析错误: {e}")
            
            print(f"从{input_path}提取了{sample_count}个样本")
            
            # 只有在有样本的情况下才保存文件
            if sample_count > 0:
                with open(output_path, 'w') as f:
                    json.dump(converted_data, f, indent=2)
                print(f"转换完成并保存到：{output_path}")
            else:
                print(f"警告：{input_path}中没有提取到任何样本，跳过保存")
                
        except FileNotFoundError:
            print(f"文件不存在：{input_path}")
            continue
        except Exception as e:
            print(f"处理文件{input_path}时发生错误: {e}")

if __name__ == "__main__":
    convert_class_json()
    convert_property_json()
    convert_memorizing_jsonl()