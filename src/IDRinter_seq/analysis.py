import torch
import yaml
import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

# 配置环境
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import IDRinter_Seq
from dataset import PPIDataModule

# 设置绘图风格
plt.rcParams['axes.unicode_minus'] = False 
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 如果有中文字体可开启

def load_config_and_model(config_path, checkpoint_path, device):
    """辅助函数：加载配置和模型"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoint from {checkpoint_path}")
    model = IDRinter_Seq.load_from_checkpoint(
        checkpoint_path,
        esm_model_path=config['model'].get('esm_path', None)
    )
    model.to(device)
    model.eval()
    return config, model

# ==============================================================================
# Task 1: 批量测试集评估
# ==============================================================================
def run_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, model = load_config_and_model(args.config, args.checkpoint, device)

    # 准备数据
    print("Preparing Test DataModule...")
    dm = PPIDataModule(
        pickle_path=config['data']['path'],
        batch_size=config['training'].get('batch_size', 16),
        num_workers=config['training'].get('num_workers', 4),
        train_ratio=config['data'].get('train_ratio', 0.8),
        seed=config['seed']
    )
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()

    print("Running inference on Test Set...")
    all_labs = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            seqs_a, seqs_b, regions_a, regions_b, labels = batch
            labels = labels.to(device)
            
            # 前向传播
            logits = model(seqs_a, seqs_b, regions_a, regions_b, return_attn=False)
            probs = torch.sigmoid(logits.squeeze(1))
            
            all_labs.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # 拼接结果
    y_true = np.concatenate(all_labs)
    y_scores = np.concatenate(all_probs)
    y_pred = (y_scores > 0.5).astype(int)

    # 结果保存目录
    save_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving evaluation results to: {save_dir}")

    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Neg', 'Pred Pos'], 
                yticklabels=['True Neg', 'True Pos'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.title('ROC Curve')
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # 3. 输出文本报告
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total Samples: {len(y_true)}\n")
        f.write(f"Accuracy: {(y_pred == y_true).mean():.4f}\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")

# ==============================================================================
# Task 2: 单样本可解释性分析 (Attention Heatmap)
# ==============================================================================
def run_interpret(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只需要加载配置和模型，不需要DataModule
    _, model = load_config_and_model(args.config, args.checkpoint, device)
    
    print("\nRunning Interpretability Analysis on Custom Sample...")
    print(f"Seq A Length: {len(args.seq_a)}")
    print(f"Seq B Length: {len(args.seq_b)}")
    print(f"Region A: {args.region_a}")
    print(f"Region B: {args.region_b}")

    # 构造输入 (Batch Size = 1)
    seqs_a = [args.seq_a]
    seqs_b = [args.seq_b]
    regions_a = [tuple(args.region_a)] # argparse read as list, convert to tuple
    regions_b = [tuple(args.region_b)]
    
    # 推理并获取 Attention
    with torch.no_grad():
        logits, attn_dict = model(seqs_a, seqs_b, regions_a, regions_b, return_attn=True)
    
    prob = torch.sigmoid(logits).item()
    pred_label = 1 if prob > 0.5 else 0
    print(f"\n=========================================")
    print(f"Prediction Probability: {prob:.4f}")
    print(f"Predicted Class: {'Positive (Interaction)' if pred_label==1 else 'Negative (No Interaction)'}")
    if args.label is not None:
        print(f"Ground Truth Label: {args.label}")
    print(f"=========================================\n")
    
    # 绘制热图
    save_dir = os.path.join(os.path.dirname(args.checkpoint), "interpretation")
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取 Attention
    # model返回的是一个list [layer1_attn, layer2_attn, ...], 需要堆叠并取平均或取最后一层
    # stack layers: (num_layers, batch, len_q, len_k)
    attn_a2b_layers = torch.stack(attn_dict['a2b'])
    attn_b2a_layers = torch.stack(attn_dict['b2a'])
    
    # 对所有层取平均，并取batch即第0个样本
    # shape: (len_q, len_k)
    attn_a2b = torch.mean(attn_a2b_layers, dim=0).cpu().numpy()[0] 
    attn_b2a = torch.mean(attn_b2a_layers, dim=0).cpu().numpy()[0]
    
    # 截取有效区域 (去除可能的 padding，虽然 batch=1 时应该没有 padding，但为了严谨)
    len_reg_a = regions_a[0][1] - regions_a[0][0]
    len_reg_b = regions_b[0][1] - regions_b[0][0]
    
    real_attn_a2b = attn_a2b[:len_reg_a, :len_reg_b]
    real_attn_b2a = attn_b2a[:len_reg_b, :len_reg_a]

    # --- Plot A2B ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(real_attn_a2b, cmap='viridis', robust=True)
    plt.title(f"Attention: Seq A Region -> Seq B Region\nProb: {prob:.3f}")
    plt.xlabel("Protein B Residue Index (in Region)")
    plt.ylabel("Protein A Residue Index (in Region)")
    plt.savefig(os.path.join(save_dir, f'{args.save_name}_attn_a2b.png'))
    plt.close()
    
    # --- Plot B2A ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(real_attn_b2a, cmap='viridis', robust=True)
    plt.title(f"Attention: Seq B Region -> Seq A Region")
    plt.xlabel("Protein A Residue Index (in Region)")
    plt.ylabel("Protein B Residue Index (in Region)")
    plt.savefig(os.path.join(save_dir, f'{args.save_name}_attn_b2a.png'))
    plt.close()
    
    print(f"Heatmaps saved to: {save_dir}")
    
    # ------------------------------------------------------------
    # Part 3: High Attention (Hotspot) Residue Extraction & Visualization
    # ------------------------------------------------------------
    # Analyze Protein A (IDR) Hotspots (Attended by B)
    visualize_hotspots(args, real_attn_b2a, 
                      seq_full=args.seq_a, 
                      region_limit=args.region_a, 
                      protein_name="Protein A", 
                      save_dir=save_dir, 
                      threshold_quantile=args.threshold)
                      
    # Analyze Protein B Hotspots (Attended by A)
    # real_attn_a2b shape: (len_a, len_b) -> Sum over A (axis 0) gives score for B residues
    visualize_hotspots(args, real_attn_a2b, 
                      seq_full=args.seq_b, 
                      region_limit=args.region_b, 
                      protein_name="Protein B", 
                      save_dir=save_dir, 
                      threshold_quantile=args.threshold)

def visualize_hotspots(args, attn_matrix, seq_full, region_limit, protein_name, save_dir, threshold_quantile=0.98):
    """
    可视化和提取高注意力残基 (Hotspots, SLiMs)
    
    Args:
        attn_matrix: Attention matrix where Columns represent the target residues we want to score.
                     Shape: (len_source, len_target)
                     我们对axis=0求和，得到每个target residue被source关注的总强度。
        seq_full: 对应的全长蛋白序列 (用于提取残基字符)
        region_limit: 该蛋白的区域范围 (start, end)
        protein_name: 蛋白名称 (e.g., "Protein A")
        threshold_quantile: 阈值分位数
    """
    print(f"\n===== {protein_name} Hotspot Analysis (Threshold: Top {100*(1-threshold_quantile):.1f}%) =====")
    
    # 1. 计算每个残基被关注的总强度
    # Matrix shape: (Source_Len, Target_Len) -> Sum(axis=0) -> (Target_Len,)
    attention_score = attn_matrix.sum(axis=0) 
    
    # 获取动态阈值
    threshold = np.quantile(attention_score, threshold_quantile)
    
    # 找出超过阈值的残基索引 (Relative to Region)
    start_idx = region_limit[0]
    hotspot_indices = np.where(attention_score > threshold)[0]
    
    # 2. 生成可视化序列字符创
    # 截取对应的实际序列片段
    target_fragment = seq_full[region_limit[0]:region_limit[1]]
    
    print(f"\n[{protein_name} Hotspots]")
    print(f"Region Limit: {region_limit} (Length: {len(target_fragment)})")
    
    # 用 ANSI Color 打印高亮序列到终端
    # 红色加粗表示 Hotspot
    terminal_output = ""
    # HTML 内容不用在这里初始化，后面生成HTML时再做
    html_content_parts = []
    
    hotspot_data = [] # 存储结果用于 CSV
    
    for i, char in enumerate(target_fragment):
        global_idx = start_idx + i
        score = attention_score[i]
        
        if i in hotspot_indices:
            # 终端高亮
            terminal_output += f"\033[91m\033[1m{char}\033[0m"
            # HTML 高亮
            html_content_parts.append(f"<span style='background-color: #ffcccc; color: red; font-weight: bold; padding: 2px;' title='Pos: {global_idx}, Score: {score:.4f}'>{char}</span>")
            
            hotspot_data.append({
                'Protein': protein_name,
                'Residue': char,
                'Region_Index': i,
                'Global_Index': global_idx,
                'Attention_Score': score
            })
        else:
            terminal_output += char
            html_content_parts.append(f"<span style='opacity: 0.6;' title='Pos: {global_idx}, Score: {score:.4f}'>{char}</span>")
            
    html_output_div = "<div style='font-family: monospace; font-size: 14px; letter-spacing: 2px; text-align: left; word-break: break-all;'>" + "".join(html_content_parts) + "</div>"
    
    print(f"Highlighted Sequence: {terminal_output}")
    print(f"\nDetailed {protein_name} Hotspot Residues:")
    print(f"{'Residue':<10} {'Global_Idx':<12} {'Region_Idx':<12} {'Attn_Score':<10}")
    print("-" * 50)
    for h in hotspot_data:
        print(f"{h['Residue']:<10} {h['Global_Index']:<12} {h['Region_Index']:<12} {h['Attention_Score']:.4f}")
        
    # 3. 保存 CSV
    df_hotspot = pd.DataFrame(hotspot_data)
    # 文件名区分蛋白
    safe_name = protein_name.replace(" ", "_")
    csv_path = os.path.join(save_dir, f'{args.save_name}_{safe_name}_hotspots.csv')
    df_hotspot.to_csv(csv_path, index=False)
    print(f"\nHotspot data saved to: {csv_path}")
    
    # 4. 保存 HTML 可视化网页
    html_path = os.path.join(save_dir, f'{args.save_name}_{safe_name}_hotspots.html')
    with open(html_path, 'w') as f:
        f.write(f"""
        <html>
        <head><title>{protein_name} Hotspot Analysis</title></head>
        <body>
            <h3>{protein_name} Hotspot Visualization</h3>
            <p><strong>Sample:</strong> {args.save_name}</p>
            <p><strong>Range:</strong> {region_limit}</p>
            <p><strong>Threshold:</strong> Top {100*(1-threshold_quantile):.1f}% Attention Score</p>
            <hr>
            {html_output_div}
            <hr>
            <p><small>* Hover over residues to see detailed scores and positions.</small></p>
        </body>
        </html>
        """)
    # print(f"Interactive HTML visualization saved to: {html_path}")
    
    # 5. 绘制 Barplot (Attention Profile)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(attention_score)), attention_score, color='skyblue', edgecolor='blue', alpha=0.7)
    # 标记阈值线
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (Top {100*(1-threshold_quantile):.0f}%)')
    # 标记 Hotspot
    plt.scatter(hotspot_indices, attention_score[hotspot_indices], color='red', zorder=5, label='Hotspots')
    
    plt.title(f"Attention Score Profile along {protein_name}")
    plt.xlabel("Residue Index (in Region)")
    plt.ylabel("Aggregated Attention Score")
    if len(target_fragment) < 100: # 只有序列较短时才显示X轴具体字符，否则太挤
        plt.xticks(
            ticks=range(len(target_fragment)), 
            labels=[f"{c}\n{i}" for i, c in enumerate(target_fragment)],
            fontsize=8
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{args.save_name}_{safe_name}_attn_profile.png'))
    plt.close()

# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDRinter Analysis Tool")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Choose analysis mode')

    # --- Mode 1: Evaluate ---
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate model performance on the entire Test Set')
    parser_eval.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser_eval.add_argument('--checkpoint', type=str, required=True, help="Path to .ckpt file")

    # --- Mode 2: Interpret ---
    parser_int = subparsers.add_parser('interpret', help='Visualize attention maps for a single custom sample')
    parser_int.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser_int.add_argument('--checkpoint', type=str, required=True, help="Path to .ckpt file")
    
    # Custom Sample Arguments
    parser_int.add_argument('--seq_a', type=str, required=True, help="Sequence of Protein A")
    parser_int.add_argument('--seq_b', type=str, required=True, help="Sequence of Protein B")
    parser_int.add_argument('--region_a', type=int, nargs=2, required=True, help="Region A start and end (e.g., 0 100)")
    parser_int.add_argument('--region_b', type=int, nargs=2, required=True, help="Region B start and end (e.g., 50 150)")
    parser_int.add_argument('--label', type=int, default=None, help="Optional ground truth label for reference")
    parser_int.add_argument('--save_name', type=str, default="sample", help="Prefix for saved heatmap files")
    parser_int.add_argument('--threshold', type=float, default=0.98, help="Quantile threshold for hotspot detection (e.g. 0.95 for top 5%%)")

    args = parser.parse_args()

    if args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'interpret':
        run_interpret(args)