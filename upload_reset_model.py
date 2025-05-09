#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将随机初始化的模型状态字典上传到Hugging Face
"""

import os
import argparse
import logging
from pathlib import Path
from huggingface_hub import HfApi, login

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="上传随机初始化的模型状态字典到Hugging Face")
    parser.add_argument('--task-name', type=str, default="ontology_class_hierarchy", help='任务名称')
    parser.add_argument('--input-dir', type=str, default="reset_models", help='输入目录')
    parser.add_argument('--repo-id', type=str, default="agaralon/acdc_reset_models", help='Hugging Face仓库ID')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API令牌')
    return parser.parse_args()

def reset_network(task: str, device, model: torch.nn.Module, use_local_first=True, custom_repo_id=None) -> None:
    """
    重置网络权重，支持从本地或Hugging Face加载模型
    
    Args:
        task: 任务名称
        device: 设备
        model: 模型
        use_local_first: 是否优先使用本地文件
        custom_repo_id: 自定义Hugging Face仓库ID
    """
    # 定义任务到文件名的映射
    task_to_filename = {
        "ioi": "ioi_reset_heads_neurons.pt",
        "tracr-reverse": "tracr_reverse_reset_heads_neurons.pt",
        "tracr-proportion": "tracr_proportion_reset_heads_neurons.pt",
        "induction": "induction_reset_heads_neurons.pt",
        "docstring": "docstring_reset_heads_neurons.pt",
        "greaterthan": "greaterthan_reset_heads_neurons.pt",
        "knowledge": "knowledge_reset_heads_neurons.pt",
        "ontology_class_hierarchy": "ontology_class_hierarchy_reset_heads_neurons.pt",
    }
    
    # 获取文件名
    filename = task_to_filename.get(task, f"{task}_reset_heads_neurons.pt")
    
    # 定义本地文件路径
    local_reset_dir = Path("reset_models")
    local_file_path = local_reset_dir / filename
    
    # 定义仓库ID
    repo_id = custom_repo_id or "agaralon/acdc_reset_models"
    
    # 尝试加载模型
    loaded = False
    
    # 如果优先使用本地文件且本地文件存在
    if use_local_first and local_file_path.exists():
        try:
            logger.info(f"从本地加载重置权重: {local_file_path}")
            reset_state_dict = torch.load(local_file_path, map_location=device)
            model.load_state_dict(reset_state_dict, strict=False)
            logger.info(f"成功从本地加载重置权重")
            loaded = True
        except Exception as e:
            logger.warning(f"从本地加载重置权重失败: {e}")
    
    # 如果本地加载失败或不优先使用本地文件，尝试从Hugging Face加载
    if not loaded:
        try:
            logger.info(f"尝试从Hugging Face ({repo_id})加载重置权重: {filename}")
            
            # 使用缓存目录，避免重复下载
            cache_dir = Path(".hf_cache")
            cache_dir.mkdir(exist_ok=True)
            
            # 下载文件
            random_model_file = huggingface_hub.hf_hub_download(
                repo_id=repo_id, 
                filename=filename,
                cache_dir=cache_dir
            )
            
            # 加载模型
            reset_state_dict = torch.load(random_model_file, map_location=device)
            model.load_state_dict(reset_state_dict, strict=False)
            logger.info(f"成功从Hugging Face加载重置权重: {filename}")
            
            # 如果本地目录不存在，创建它
            if not local_reset_dir.exists():
                local_reset_dir.mkdir(parents=True, exist_ok=True)
            
            # 将下载的文件保存到本地，以便下次使用
            if not local_file_path.exists():
                import shutil
                shutil.copy(random_model_file, local_file_path)
                logger.info(f"已将模型文件保存到本地: {local_file_path}")
                
            loaded = True
        except Exception as e:
            logger.warning(f"从Hugging Face加载重置权重失败: {e}")
    
    # 如果所有加载尝试都失败，使用随机初始化
    if not loaded:
        logger.warning("无法加载重置权重，使用随机初始化")
        # 随机初始化模型权重
        for param in model.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

def upload_reset_model(task_name, input_dir, repo_id, token):
    """
    上传随机初始化的模型状态字典到Hugging Face
    
    Args:
        task_name: 任务名称
        input_dir: 输入目录
        repo_id: Hugging Face仓库ID
        token: Hugging Face API令牌
    """
    # 登录Hugging Face
    login(token=token)
    
    # 创建API对象
    api = HfApi()
    
    # 构建文件路径
    input_path = Path(input_dir)
    file_name = f"{task_name}_reset_heads_neurons.pt"
    file_path = input_path / file_name
    
    # 检查文件是否存在
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return False
    
    # 上传文件
    logger.info(f"正在上传文件 {file_path} 到 {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model"
        )
        logger.info(f"文件上传成功!")
        return True
    except Exception as e:
        logger.error(f"上传文件时出错: {e}")
        return False

def main():
    """主函数"""
    args = parse_args()
    success = upload_reset_model(
        task_name=args.task_name,
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        token=args.token
    )
    
    if success:
        logger.info(f"随机初始化的模型状态字典已成功上传到 {args.repo_id}")
    else:
        logger.error("上传失败")

if __name__ == "__main__":
    main()