#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成随机初始化的模型状态字典，用于知识任务的零消融实验
"""

import os
import torch
import argparse
import logging
from pathlib import Path
from transformers import GPT2Config, GPT2Model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成随机初始化的模型状态字典")
    parser.add_argument('--model-name', type=str, default="gpt2-medium", help='模型名称')
    parser.add_argument('--output-dir', type=str, default="reset_models", help='输出目录')
    parser.add_argument('--task-name', type=str, default="ontology_class_hierarchy", help='任务名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def generate_reset_model(model_name, output_dir, task_name, seed=42):
    """
    生成随机初始化的模型状态字典
    
    Args:
        model_name: 模型名称
        output_dir: 输出目录
        task_name: 任务名称
        seed: 随机种子
    """
    # 设置随机种子
    torch.manual_seed(seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取模型配置
    logger.info(f"获取模型 {model_name} 的配置...")
    if model_name == "gpt2":
        config = GPT2Config()
    elif model_name == "gpt2-medium":
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16
        )
    elif model_name == "gpt2-large":
        config = GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20
        )
    elif model_name == "gpt2-xl":
        config = GPT2Config(
            n_embd=1600,
            n_layer=48,
            n_head=25
        )
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    # 创建随机初始化的模型
    logger.info("创建随机初始化的模型...")
    model = GPT2Model(config)
    
    # 获取模型状态字典
    state_dict = model.state_dict()
    
    # 保存模型状态字典
    output_file = output_path / f"{task_name}_reset_heads_neurons.pt"
    logger.info(f"保存模型状态字典到 {output_file}...")
    torch.save(state_dict, output_file)
    
    logger.info("完成!")
    return output_file

def main():
    """主函数"""
    args = parse_args()
    output_file = generate_reset_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        task_name=args.task_name,
        seed=args.seed
    )
    logger.info(f"随机初始化的模型状态字典已保存到: {output_file}")

if __name__ == "__main__":
    main()