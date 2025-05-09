# 本体知识数据集主程序
# 基于acdc/main.py修改，专门用于处理本体知识数据集

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import huggingface_hub

# 导入ACDC相关模块
from acdc.knowledge.ontology_dataset import OntologyDataset
from acdc.knowledge.utils import get_all_knowledge_things
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import (
    make_nd_dict,
    cleanup,
    Edge,
    EdgeType,
)
# 导入可视化相关模块
from acdc.acdc_graphics import show

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用梯度计算，因为我们只进行推理
torch.autograd.set_grad_enabled(False)

def reset_network(task: str, device, model: torch.nn.Module) -> None:
    """
    重置网络权重
    
    Args:
        task: 任务名称
        device: 设备
        model: 模型
    """
    try:
        # 首先尝试从本地加载
        local_reset_dir = Path("reset_models")
        local_filename = f"{task}_reset_heads_neurons.pt"
        local_file_path = local_reset_dir / local_filename
        
        if local_file_path.exists():
            logger.info(f"从本地加载重置权重: {local_file_path}")
            reset_state_dict = torch.load(local_file_path, map_location=device)
            model.load_state_dict(reset_state_dict, strict=False)
            logger.info(f"成功从本地加载重置权重")
            return
            
        # 如果本地文件不存在，尝试从Hugging Face加载
        filename = {
            "ioi": "ioi_reset_heads_neurons.pt",
            "tracr-reverse": "tracr_reverse_reset_heads_neurons.pt",
            "tracr-proportion": "tracr_proportion_reset_heads_neurons.pt",
            "induction": "induction_reset_heads_neurons.pt",
            "docstring": "docstring_reset_heads_neurons.pt",
            "greaterthan": "greaterthan_reset_heads_neurons.pt",
            "knowledge": "knowledge_reset_heads_neurons.pt",
            "ontology_class_hierarchy": "ontology_class_hierarchy_reset_heads_neurons.pt",
        }.get(task, "knowledge_reset_heads_neurons.pt")
        
        logger.info(f"尝试从Hugging Face加载重置权重: {filename}")
        random_model_file = huggingface_hub.hf_hub_download(repo_id="agaralon/acdc_reset_models", filename=filename)
        reset_state_dict = torch.load(random_model_file, map_location=device)
        model.load_state_dict(reset_state_dict, strict=False)
        logger.info(f"成功从Hugging Face加载重置权重: {filename}")
    except Exception as e:
        logger.warning(f"无法加载重置权重: {e}")
        logger.info("使用随机初始化进行重置")
        # 随机初始化模型权重
        for param in model.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="本体知识数据集的ACDC实验")
    
    # 基本参数
    parser.add_argument('--threshold', type=float, default=0.1, help='ACDC阈值')
    parser.add_argument('--device', type=str, default="cuda", help='使用的设备')
    parser.add_argument('--model-name', type=str, default="gpt2", help='模型名称')
    parser.add_argument('--model-path', type=str, default="gpt2", help='模型路径')
    parser.add_argument('--data-path', type=str, default="", help='数据路径')
    parser.add_argument('--relation-name', type=str, default="", help='关系名称')
    parser.add_argument('--relation-reverse', type=str, default="False", help='是否反转关系')
    parser.add_argument('--num-examples', type=int, default=10, help='使用的样本数量')
    parser.add_argument('--metric', type=str, default="match_nll", help='使用的评估指标')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max-num-epochs', type=int, default=100000, help='最大迭代次数')
    parser.add_argument('--zero-ablation', action='store_true', help='使用零消融')
    parser.add_argument('--indices-mode', type=str, default="normal", help='索引模式')
    parser.add_argument('--names-mode', type=str, default="normal", help='名称模式')
    parser.add_argument('--first-cache-cpu', type=str, default="True", help='是否将第一个缓存放在CPU上')
    parser.add_argument('--second-cache-cpu', type=str, default="True", help='是否将第二个缓存放在CPU上')
    parser.add_argument('--abs-value-threshold', action='store_true', help='使用绝对值阈值')
    # 添加可视化相关参数
    parser.add_argument('--visualize', action='store_true', help='是否可视化知识电路')
    parser.add_argument('--output-dir', type=str, default="acdc/ontology_results", help='输出目录')
    parser.add_argument('--colorscheme', type=str, default="Pastel2", help='可视化配色方案')
    
    return parser.parse_args()

def visualize_circuit(correspondence, edges, output_dir, relation_name, model_name, colorscheme="Pastel2"):
    """
    可视化知识电路并保存结果
    
    Args:
        correspondence: ACDC对应关系
        edges: 找到的边
        output_dir: 输出目录
        relation_name: 关系名称
        model_name: 模型名称
        colorscheme: 配色方案
    """
    # 创建输出目录
    output_path = Path(output_dir) / model_name / relation_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 将边添加到对应关系中
    for edge in edges:
        parent_node = correspondence.graph[edge.parent_name][edge.parent_index]
        child_node = correspondence.graph[edge.child_name][edge.child_index]
        correspondence.add_edge(
            parent_node=parent_node,
            child_node=child_node,
            edge=edge,
            safe=False
        )
    
    # 可视化并保存
    logger.info(f"正在可视化知识电路...")
    
    # 保存中间图
    show(
        correspondence=correspondence,
        fname=str(output_path / "intermediate_graph.pdf"),
        colorscheme=colorscheme,
        show_placeholders=False,
        edge_type_colouring=True
    )
    
    # 保存最终图
    show(
        correspondence=correspondence,
        fname=str(output_path / "final_graph.pdf"),
        colorscheme=colorscheme,
        show_placeholders=False,
        edge_type_colouring=False
    )
    
    # 保存带占位符的图
    show(
        correspondence=correspondence,
        fname=str(output_path / "full_graph.pdf"),
        colorscheme=colorscheme,
        show_placeholders=True,
        edge_type_colouring=False
    )
    
    logger.info(f"知识电路可视化完成，结果保存在 {output_path}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 将字符串转换为布尔值
    relation_reverse = args.relation_reverse.lower() == "true"
    first_cache_cpu = args.first_cache_cpu.lower() == "true"
    second_cache_cpu = args.second_cache_cpu.lower() == "true"
    
    # 加载本体知识数据集
    logger.info("加载本体知识数据集...")
    
    # 获取知识数据
    all_knowledge_things = get_all_knowledge_things(
        num_examples=args.num_examples,
        device=device,
        model=args.model_name,
        model_path=args.model_path,
        knowledge_type="ontology",
        data_path=args.data_path,
        relation_name=args.relation_name,
        reverse=relation_reverse,
        data_seed=args.seed,
        metric_name=args.metric,
    )
    
    # 提取必要的数据
    tl_model = all_knowledge_things.tl_model
    validation_metric = all_knowledge_things.validation_metric
    validation_data = all_knowledge_things.validation_data
    validation_patch_data = all_knowledge_things.validation_patch_data
    test_metrics = all_knowledge_things.test_metrics
    test_data = all_knowledge_things.test_data
    test_patch_data = all_knowledge_things.test_patch_data
    
    # 创建ACDC实验
    logger.info("创建ACDC实验...")
    
    # 如果需要，重置网络
    if args.zero_ablation:
        logger.info("使用零消融...")
        reset_network(args.relation_name, device, tl_model)
    
    # 创建ACDC实验
    experiment = TLACDCExperiment(
        model=tl_model,
        ds=validation_data,
        ref_ds=validation_patch_data,
        threshold=args.threshold,
        metric=validation_metric,
        second_metric=None,
        verbose=True,
        online_cache_cpu=first_cache_cpu,
        corrupted_cache_cpu=second_cache_cpu,
        zero_ablation=args.zero_ablation,
        abs_value_threshold=args.abs_value_threshold,
        indices_mode=args.indices_mode,
        names_mode=args.names_mode,
    )
    
    # 运行ACDC实验
    logger.info("运行ACDC实验...")
    for i in tqdm(range(args.max_num_epochs)):
        if experiment.step():
            break
    
    # 获取结果
    logger.info("获取实验结果...")
    edges = []
    for (child_name, child_index, parent_name, parent_index), edge in experiment.corr.all_edges().items():
        if edge.present:
            edges.append(edge)
    
    # 打印结果
    logger.info(f"找到 {len(edges)} 条边")
    for i, edge in enumerate(edges):
        logger.info(f"边 {i+1}: {edge}")
    
    # 测试结果
    logger.info("测试结果...")
    test_metric = test_metrics[args.metric]
    
    # 在测试数据上评估
    with torch.no_grad():
        # 原始模型性能
        original_outputs = tl_model(test_data)
        original_metric = test_metric(original_outputs).item()
        logger.info(f"原始模型在测试集上的性能: {original_metric:.4f}")
        
        # 重置模型
        reset_network(args.relation_name, device, tl_model)
        
        # 添加找到的边
        for edge in edges:
            parent_node = experiment.corr.graph[edge.parent_name][edge.parent_index]
            child_node = experiment.corr.graph[edge.child_name][edge.child_index]
            experiment.corr.add_edge(
                parent_node=parent_node,
                child_node=child_node,
                edge=edge,
                safe=False
            )
        
        # 评估添加边后的模型性能
        circuit_outputs = tl_model(test_data)
        circuit_metric = test_metric(circuit_outputs).item()
        logger.info(f"电路模型在测试集上的性能: {circuit_metric:.4f}")
        
        # 计算性能恢复率
        recovery_rate = circuit_metric / original_metric * 100
        logger.info(f"性能恢复率: {recovery_rate:.2f}%")
    
    # 可视化知识电路
    if args.visualize:
        visualize_circuit(
            correspondence=experiment.corr,
            edges=edges,
            output_dir=args.output_dir,
            relation_name=args.relation_name,
            model_name=args.model_name,
            colorscheme=args.colorscheme
        )
    
    logger.info("实验完成!")
    
    # 清理
    cleanup()
    
    return edges

if __name__ == "__main__":
    main()