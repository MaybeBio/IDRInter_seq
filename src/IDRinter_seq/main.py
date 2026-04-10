import torch
import yaml # 读取YAML配置文件
import argparse # 解析命令行参数
import lightning as L 
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping  # 保存模型, 监控学习率, 早停
from lightning.pytorch.loggers import TensorBoardLogger # TensorBoard日志记录
import logging # 日志记录
import sys
import os
import datetime # 日志文件版本时间戳

# Configure logging
# 配置日志记录
logging.basicConfig(
    level=logging.INFO, # 输出INFO级别及以上日志
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # 日志信息输出格式: 时间-名称-级别-信息
    handlers=[logging.StreamHandler(sys.stdout)] # 日志处理器: 输出到控制台/终端, 后面还可以添加文件处理器等
)
logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

# Ensure we can import from local modules
# 确保可以导入本地模块, 直接将main文件所在目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入本地模块
from model import IDRinter_Seq
from dataset import PPIDataModule


def main(args):
    """ 
    Description
    -----------
    Main function to set up and run the training process for the IDRinter_Seq model.
    模型训练主函数

    Args
    ----
    args : argparse.Namespace
        Command line arguments parsed by argparse. 命令行参数    
    
    """
    
    # 1. Load configuration from YAML file
    # 从YAML配置文件加载配置
    logger.info(f"Loading configuration from {args.config}")
    # 打开命令行参数传入的 yaml 文件路径
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f) # 读取并解析 yaml 文件内容为字典
        
    # Set seed for reproducibility / 设置随机种子
    # Lightning提供的工具，一次性固定 numpy, torch, python random 的所有种子
    L.seed_everything(config.get('seed', 2026)) 
    
    # 2. Prepare DataModule
    # 准备数据模块
 
    logger.info("Preparing DataModule...")
    data_cfg = config['data']         # 提取yaml中的data部分配置
    training_cfg = config['training'] # 提取yaml中的training部分配置
    
    # 实例化数据模块
    dm = PPIDataModule(
        pickle_path = data_cfg['path'],
        batch_size = training_cfg.get('batch_size', 16),
        num_workers = training_cfg.get('num_workers', 10),
        train_ratio = data_cfg.get('train_ratio', 0.8), # 训练集比例, 一次性划分
        seed = config.get('seed', 2026) # 传种子进dataset，保证划分train/val时也是固定的
    )
    
    # 3. Model initialization 
    # 模型初始化
    logger.info("Initializing model...")
    # 提取yaml中的model部分配置
    model_cfg = config['model'] 
    
    # 实例化LightningModule, IDRinter_Seq模型, 包括初始化ESM-2和Decoder网络
    model = IDRinter_Seq(
        esm_model_path=model_cfg.get('esm_path', None), # 如果有本地权重路径则加载，没有则None
        num_decoder_layers=model_cfg.get('layers', 2),  # 解码器层数，默认2层
        embed_dim=model_cfg.get('embed_dim', 1280),     # ESM-2 650M 的维度是1280
        num_heads=model_cfg.get('heads', 8),            # 注意力头数
        dropout=model_cfg.get('dropout', 0.1),          # 防止过拟合的dropout率
        learning_rate=float(training_cfg['lr'])         # 学习率从training配置里拿
    )
    
    # 4. Callbacks setup
    # 回调函数设置
    logger.info("Setting up callbacks...")
    logging_cfg = config['logging']
    
    # 确保总日志目录存在
    os.makedirs(logging_cfg['save_dir'], exist_ok=True)
    
    # Create timestamp for unique versioning
    # 创建时间戳用于唯一版本控制
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version = timestamp
    
    # Define experiment directory
    # 定义实验目录, logs/实验名/20260202_143000/
    experiment_dir = os.path.join(logging_cfg['save_dir'], logging_cfg['name'], version)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Add FileHandler to logger
    # 在 实验专属目录 创建好之后，我们动态创建一个 FileHandler 指向里面的 training.log
    file_handler = logging.FileHandler(os.path.join(experiment_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    # 将这个handler加到根logger上
    # 从这一行开始，所有的日志（包括Dataset里的日志）不仅会打印在屏幕上，还会自动写入 training.log 文件, 方便后续查看
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Log file saved to: {os.path.join(experiment_dir, 'training.log')}")
    
    
    # Model Checkpoint callback
    # 模型检查点回调函数
    # Construct checkpoint path: save_dir/name/version/checkpoints
    # 模型权重文件保存在实验目录下的 checkpoints 子文件夹
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    
    callbacks = [
        # ModelCheckpoint: 负责保存模型
        ModelCheckpoint(
            dirpath=ckpt_dir,
            # 文件名格式：epoch-损失-f1分数。
            # 不用加载模型，光看文件名就知道这个模型在验证集上效果好不好
            filename='{epoch}-{val_loss:.4f}-{val_f1:.4f}', 
            monitor='val_loss', # 监控验证集损失
            mode='min',         # 损失越小越好
            save_top_k=3,       # 只保留效果最好的3个
            save_last=True      # 额外保存一个 last.ckpt，方便中断后恢复
        ),
        # LR Monitor: 记录学习率变化
        # 可以设置为按epoch或按step记录，这里选择按step记录, 1个epoch里batch_size个step, 每1个batch更新1个step
        LearningRateMonitor(logging_interval='step'),
        
        # Early Stopping: 早停机制
        EarlyStopping(
            monitor='val_loss', # 盯着验证集损失
            patience=training_cfg.get('patience', 10), # 如果连续10个epoch损失都不下降
            mode='min' # 且目标是越小越好
        ) # 就会强制停止训练
    ]
    
    # 5. Logger setup
    # 配置TensorBoard日志记录器
    l_logger = TensorBoardLogger(
        save_dir=logging_cfg['save_dir'],
        name=logging_cfg['name'],
        version=version # 显式传入前面生成的时间戳version，确保TensorBoard日志也存放在同一个目录下
    )
    
    # 6. Trainer / 初始化Trainer (指挥官)
    logger.info("Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=training_cfg['epochs'], # 最大训练轮数
        accelerator='auto', # 智能检测：有GPU用GPU，没GPU用CPU
        devices='auto',     # 自动选择可用设备数量
        callbacks=callbacks, # 装载上面定义的回调列表
        logger=l_logger,     # 装载TensorBoard记录器
        log_every_n_steps=10, # 每10个batch记录一次Loss，避免日志文件过大
        precision=training_cfg.get('precision', '16-mixed'), # 半精度训练,
        accumulate_grad_batches=training_cfg.get('accumulate_grad_batches', 2), # 从配置读取
    )
    
    # 7. Start Training / 开始训练
    logger.info("Starting Training...")
    
    # 整个流程：
    # 1. 调用 dm.setup('fit') 加载数据
    # 2. 将 model 移动到 GPU
    # 3. 开始 Epoch 循环 -> Training Step -> Validation Step -> 保存模型
    trainer.fit(model, datamodule=dm)
    
    logger.info("Training Complete!")
    
# Entry point
# 程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IDRinter_Seq Model")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    # 执行主函数
    main(args)