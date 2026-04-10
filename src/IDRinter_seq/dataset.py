import torch 
from torch.utils.data import Dataset, DataLoader
from typing import *
from sklearn.model_selection import train_test_split # 用于划分训练集和验证集
import logging # 日志记录 
import pandas as pd
import lightning as L
import random # 用于设置随机种子2026

# 实例化logger, 日志记录
# __name__代表当前模块名称, 如果是主程序运行, 则其值为'__main__'; 
# 如果该dataset.py被其他模块导入, 则其值为被导入的模块名(dataset), 便于区分日志来源
logger = logging.getLogger(__name__)


class PPIDataset(Dataset):
    """  
    Description
    -----------
    PPI dataset for IDRinter_seq model training and evaluation.
    
    """
    
    def __init__(self, ppi_df: pd.DataFrame):
        """ 
        Description
        -----------
        initialize the dataset
        初始化互作PPI数据集, 从pd.DataFrame转化为pytorch Dataset格式
        
        
        Args
        ----
            ppi_df: pd.DataFrame, 包含PPI数据的数据框, 包含seqA/seqB/label数据
        """
        
        # 将pd.dataframe每一行转化为字典, 存储在列表中, 便于后续按索引访问每一行
        self.ppi_df = ppi_df.to_dict('records')
        
    def __len__(self):
        """ 
        Description
        -----------
        return the length of the dataset
        返回数据集的长度(样本数量)
        
        Returns
        -------
            int, 数据集的长度(样本数量)
        """
        # 每一行一字典
        return len(self.ppi_df)
        
    def __getitem__(self, idx):
        """     
        Description
        -----------
        按照索引获取数据集中的任意1个样本, 样本格式为1个元组(seqA, seqB, regionA, regionB, label)
        
        Args
        ----
            idx: int, 样本索引, 0-based
        """    
        
        item = self.ppi_df[idx]
        return (
            item['seq_a'],
            item['seq_b'],
            item['region_a'],
            item['region_b'],
            torch.tensor(item['label'], dtype=torch.float32)
        )
        
        
def collate_fn(batch):
    """   
    Description
    -----------
    Custom collate function for DataLoader to handle batching of variable-length sequences.
    自定义的批处理函数, 用于DataLoader处理变长序列的批处理, 
    主要是将1批数据中每一行样本的序列、区域坐标信息保存为list, 将标签保存为tensor,
    便于后续在dataloader中调用按批次加载数据
    
    Args
    ----
        batch: list of tuples, 每个元组为一个样本, 包含(seqA, seqB, regionA, regionB, label)
    
    Returns
    -------
        seq_a_list: List[str], 包含批次中所有样本的seqA序列的列表
        seq_b_list: List[str], 包含批次中所有样本的seqB序列的列表
        region_a_list: List[Tuple[int, int]], 包含批次中所有样本的regionA区域坐标的列表
        region_b_list: List[Tuple[int, int]], 包含批次中所有样本的regionB区域坐标的列表
        labels: Tensor of shape (batch_size,), 包含批次中所有样本标签的1维张量
    """
    
    seq_a_list = []
    seq_b_list = []
    region_a_list = []
    region_b_list = []
    labels = []
    
    for seq_a, seq_b, region_a, region_b, label in batch:
        seq_a_list.append(seq_a)
        seq_b_list.append(seq_b)
        region_a_list.append(region_a)
        region_b_list.append(region_b)
        labels.append(label)
        
    labels = torch.stack(labels)  # 将标签列表转换1维张量
    
    return seq_a_list, seq_b_list, region_a_list, region_b_list, labels


class PPIDataModule(L.LightningDataModule):
    """  
    Description
    -----------
    Initialize PPI DataModule for PyTorch Lightning.
    Handles loading raw data from pickle, splitting into train/val sets,
    用于PyTorch Lightning的PPI数据模块初始化, 管理数据加载/划分与预处理等操作
    
    """
    
    def __init__(self, pickle_path:str, 
                 batch_size: int = 32, 
                 num_workers: int = 10, 
                 train_ratio: float = 0.8, 
                 seed: int = 2026):
        """   
        Description
        -----------
        initialize the DataModule
        
        Args
        -----
            pickle_path: str, path to the pickle file containing PPI data
            batch_size: int, batch size for DataLoader
            num_workers: int, number of worker processes for data loading
            train_ratio: float, ratio of training set size to the entire dataset
            seed: int, random seed for reproducibility

        
        Notes
        -----
        - 1, num_workers: lscpu查看物理核心数 cores all sockets, 考虑到其他进程占用, 一般设置为物理核心数-1    
        - 2, batch_size: 根据显存大小调整, 一般为2的幂次方, 如32/64/128等, 此处人为限制为至少每个epoch训练50个batch左右
        """
        
        super().__init__()
        self.pickle_path = pickle_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 划分数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 防止多次初始化DataLoader的变量
        self._is_setup = False 
        
    def setup(self, stage: Optional[str] = None):
        """ 
        Description
        -----------
        setup the datasets for training, validation, and testing
        初始化训练/验证/测试数据集
        
        Args
        ----
            stage: Optional[str], stage of setup, can be 'fit', 'validate', 'test', or 'predict'
            
        Notes
        ------
        - 1, _is_setup: 确保只初始化1次数据集, 避免重复加载数据; 比如说在训练和验证阶段都会调用setup方法, 
        确保验证阶段model不会看到训练阶段的数据
        """
        
        # If datasets are already set up, skip re-initialization
        if self._is_setup:
            return
        
        # 1. Load Data 加载数据
        try:
            ppi_df = pd.read_pickle(self.pickle_path)
            logger.info(f"Loaded data from {self.pickle_path}, shape: {ppi_df.shape}")
        except Exception as e:
            logger.error(f"Error loading pickle file: {e}")
            raise e
        
        # 2. Data Preprocessing 数据预处理
        # 原始列名: IDR_id	IDR_sequence	IDR_start	IDR_end	Ord_id	Ord_sequence	Ord_start	Ord_end	label
        # Goal: normalize columns to 'seq_a', 'region_a', 'seq_b', 'region_b', 'label'
        # 标准化列名
        ppi_all = []
        
        for idx, row in ppi_df.iterrows():
            # 提取必要字段, 并重命名为标准列名
            label = float(row['label'])
            
            # 区域坐标以元组形式存储, 并转化为0-based索引
            # 参考: IDRInter/src/IDRinter_seq/sequence_embedding.py中的extract_region_embeddings函数, 我们假定输入为0-based区域坐标
            region_a = (int(row['IDR_start']) - 1, int(row['IDR_end']) - 1)
            region_b = (int(row['Ord_start']) - 1, int(row['Ord_end']) - 1)

            ppi_all.append(
                {
                    'seq_a': row['IDR_sequence'],
                    'region_a': region_a,
                    'seq_b': row['Ord_sequence'],
                    'region_b': region_b,
                    'label': label
                }
            )            
        
        
        logger.info(f"Preprocessed PPI data, total samples: {len(ppi_all)}")
        
        # 3. Split Data 划分数据集
        ppi_all_df = pd.DataFrame(ppi_all)
        labels = ppi_all_df['label'].values
        
        # 划分逻辑: 80%训练集, 10%验证集, 10%测试集
        ppi_train_df, ppi_valid_test_df = train_test_split(
            ppi_all_df,
            train_size = self.train_ratio,
            random_state = self.seed,
            shuffle = True,
            stratify=labels  # 类别在分层抽样中保持不变
        )
        
        # 再进一步划分验证集和测试集
        valid_test_labels = ppi_valid_test_df['label'].values
        ppi_valid_df, ppi_test_df = train_test_split(
            ppi_valid_test_df,
            train_size = 0.5,
            random_state = self.seed,
            shuffle = True,
            stratify=valid_test_labels  # 类别在分层抽样中保持不变
        )
        
        logger.info(f"Split data into train/val/test sets: "
                    f"train size: {ppi_train_df.shape}, "
                    f"val size: {ppi_valid_df.shape}, "
                    f"test size: {ppi_test_df.shape}")
        
        # 4. Create Dataset objects 创建Dataset对象
        # 主要就是为不同stage创建不同的数据集对象
        if stage == 'fit' or stage is None:
            self.train_dataset = PPIDataset(ppi_train_df)
            self.val_dataset = PPIDataset(ppi_valid_df)
            logger.info(f"Setup 'fit': Train size={len(self.train_dataset)}, Val size={len(self.val_dataset)}")
            
        if stage == 'test' or stage is None:
            self.test_dataset = PPIDataset(ppi_test_df)
            logger.info(f"Setup 'test': Test size={len(self.test_dataset)}")
            
        # Mark as setup completed
        self._is_setup = True
        
        
    # DataLoader methods for train/val/test
    # 前面setup方法初始化数据集对象, 此处将数据集传入到DataLoader中      
    def train_dataloader(self):
        """
        Description
        -----------
        return DataLoader for training dataset
        返回训练集的DataLoader
        
        """  
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True # 锁页内存, 加速数据转移到GPU, 设备好的可以开启
        )
    
    def val_dataloader(self):
        """ 
        Description
        -----------
        return DataLoader for validation dataset
        返回验证集的DataLoader
        
        """  
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True # 锁页内存, 加速数据转移到GPU
        )
        
    def test_dataloader(self):
        """
        Description
        -----------
        return DataLoader for testing dataset
        返回测试集的DataLoader
        
        """           
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )