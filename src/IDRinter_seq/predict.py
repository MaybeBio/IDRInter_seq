import argparse
import yaml # pip install pyyaml
import torch
import sys
import os
import pandas as pd
import logging

# Configure logging
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure we can import from local modules
# 确保可以导入当前目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import IDRinter_Seq

def predict(args):
    """
    Description
    -----------
    Run prediction on provided sequences or dataset using a trained checkpoint.
    运行推理/预测任务
    """
    
    # 1. Load Config / 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Model / 加载模型
    logger.info(f"Loading model from {args.checkpoint}...")
    # load_from_checkpoint separates hparams automatically if saved
    # 从Checkpoint加载模型, 这里IDRinter_Seq.load_from_checkpoint会自动处理超参数
    model = IDRinter_Seq.load_from_checkpoint(
        args.checkpoint,
        esm_model_path=config['model'].get('esm_path', None)
    )
    model.eval() # 设置为评估模式
    
    # Move to GPU if available
    # 如果有GPU则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on {device}")

    # 3. Prepare Input / 准备输入数据
    # Example: predict for a single pair passed via CLI
    # 示例: 通过命令行参数预测单对序列的互作
    
    # 确保提供了必要的参数
    if args.seq_a and args.seq_b:
        logger.info("Predicting for input sequences...")
        
        seq_a = args.seq_a
        seq_b = args.seq_b
        
        # region format parsing: "start-end" (1-based to 0-based conversation)
        # 解析区域格式 (例如 "10-20"), 并将1-based索引转换为0-based索引(模型内部使用)
        ra = list(map(int, args.region_a.split('-')))
        rb = list(map(int, args.region_b.split('-')))
        
        # Convert to 0-based for model input
        region_a = (ra[0]-1, ra[1]-1)
        region_b = (rb[0]-1, rb[1]-1)
        
        # Create batch of size 1
        seqs_a = [seq_a]
        seqs_b = [seq_b]
        regions_a = [region_a]
        regions_b = [region_b]
        
        # Inference
        # 推理
        with torch.no_grad():
            logits = model(seqs_a, seqs_b, regions_a, regions_b)
            prob = torch.sigmoid(logits).item() # Logits -> Probability
            
        logger.info("-" * 30)
        logger.info(f"Prediction Result / 预测结果:")
        logger.info(f"Probability (Interaction) / 互作概率: {prob:.4f}")
        logger.info(f"Class / 类别: {'Interacting (互作)' if prob > 0.5 else 'Non-interacting (不互作)'}")
        logger.info("-" * 30)

    else:
        logger.warning("No input sequences provided. Use --seq_a --seq_b --region_a --region_b")
        # Extend here for batch file prediction if needed
        # 如果需要批量预测文件, 可以在这里扩展逻辑, 读取csv/pkl然后循环推理

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict IDR-DomINTER / IDR-有序结构域 互作预测")
    parser.add_argument('config', type=str, help='Path to config file / 配置文件路径')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint .ckpt file / 模型权重文件路径')
    
    parser.add_argument('--seq_a', type=str, help='Protein A Sequence (IDR) / 序列A')
    parser.add_argument('--region_a', type=str, help='Region A start-end (e.g. 10-20) / IDR区域')
    parser.add_argument('--seq_b', type=str, help='Protein B Sequence (Ordered) / 序列B')
    parser.add_argument('--region_b', type=str, help='Region B start-end (e.g. 50-60) / 结构域区域')
    
    args = parser.parse_args()
    predict(args)
