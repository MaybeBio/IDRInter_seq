# feature/sequence_embed.py
# Task: 分别提取两条序列的embedding

import torch 
import torch.nn as nn
import esm # pip install fair-esm
from typing import *

class SeqEmbedModel(nn.Module):
    """ 
    Description
    -----------
    Wrapper for ESM-2 protein language model to generate residue and sequence embeddings.    
    序列embedding提取模块
    
    Notes
    -----
    - 1, ⚠️ DeepPCT Contrast:
        - 1. DeepPCT (sequence_embedding.py):
        Focus: Intra-protein site-site interaction (residue pairs within one protein).
        Output: Specific windows around sites (ResidueEmbeddings) and self-attention maps between sites (ResiduePairEmbeddings).
    
        - 2. This Implementation (PPI):
        Focus: Inter-protein interaction (Protein A vs Protein B).
        Output: Full sequence representations to be fed into Cross-Attention layers.
        Note: While DeepPCT uses ESM self-attention maps as features for residue pairs, 
        for separate proteins A and B, ESM only provides intra-protein attention. 
        Inter-protein interaction is learned via the Cross-Attention layers in model.py.
         
    Todos
    -----
    - 1, 🧾 We use esm2_t33_650M_UR50D pretrained model here, but we will use larger models like ESM-3 in the future , refer to https://huggingface.co/facebook/esm2_t33_650M_UR50D
    """
    
    def __init__(self, esm_model_path: str = None):
        """  
        Description
        -----------
        Load ESM-2 pretrained model. Default: esm2_t33_650M_UR50D
        加载预训练的ESM-2模型(650M) 
        
        Args
        ----
            esm_model_path: Path to local ESM-2 weights. If None, downloads from hub.
        """
        super().__init__()
        
        # 加载预训练的ESM-2模型, 可以预先下载保存好权重文件, 如果没有就重新下载
        # 下载参考https://github.com/facebookresearch/esm#available-models
        # 1️⃣重新下载的话每次都很麻烦, 但是会避免一些版本兼容性问题
        # 2️⃣如果要加载预先下载的模型, 也就是使用load_model_and_alphabet_local函数的话, 需要在/miniconda3/envs/ml_base/lib/python3.13/site-packages/esm/pretrained.py 
        # 也就是该model所在环境中该包源码文件pretrained.py中, 在line  70处修改torch.load的参数, 增加weights_only=False, 否则会报错
        # 如果是预先下载保存的话, 要下载2个文件, 一个是模型权重esm2_t33_650M_UR50D.pt, 另一个是回归权重(contact-regression) esm2_t33_650M_UR50D-contact-regression.pt
        
        # load ESM-2 here
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model_path) if esm_model_path else esm.pretrained.esm2_t33_650M_UR50D()
        
        # init batch converter which will convert raw sequence strings to model input tensors needed in training/inference
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze model for inference/feature extraction
        self.esm_model.eval() 
        
        # Determining embedding dimension from the model args
        self.embed_dim = self.esm_model.embed_dim

    def get_embedding_dim(self) -> int:
        """
        Description
        -----------
        Helper to get embedding dimension for downstream layers
        获取embedding维度
        """
        return self.embed_dim

    def generate_protein_embeddings(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """   
        Description
        -----------
        Forward process of ESM-2 model to generate sequence embedding, Compute embeddings for a list of sequences.
        预训练model前向推理, 提取序列embedding
        
        Args
        ----
            seqs: List[str], List of protein sequences
            
        Returns
        -------
            token_repr: [Batch, Max_Len, Dim] - Per-residue representations (includes BOS/EOS) for each sequence
            attentions: [Batch, Layers, Heads, Max_Len, Max_Len] - Attention maps (optional usage)
        """   
        
        # 1. Prepare Batch Data
        # refer https://github.com/facebookresearch/esm?tab=readme-ov-file#getting-started-with-this-repo-
        # prepare data like (lablel, str): [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),("protein3",  "K A <mask> I S Q"), ...]
        data = [ (str(i), seq) for i, seq in enumerate(seqs) ]
        # unpack the batch data, (label, str) same above in data, batch_tokens is the tensor input of model that we need (the embedding we need)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # 2. move inputs to the same device of model
        # 将输入数据移动到和模型相同的设备上, 首先是获取model所在的设备, next获取迭代器的初始值, 所在的device
        device = next(self.esm_model.parameters()).device
        batch_tokens = batch_tokens.to(device)
        
        # 3. Inference, Extract pre-residue representations 
        with torch.no_grad():
            # check ?self.esm_model to see available parameters, we need the last layer's output (.num_layers to get the last layer's index)
            # results here is a dict, with keys ['logits', 'representations', 'attentions', ❌'contacts'], we need 'representations' and 'attentions' keys
            results = self.esm_model(batch_tokens, 
                                     repr_layers=[self.esm_model.num_layers],
                                     need_head_weights=False)
            # 这里need_head_weights=False表示不需要返回分类头的权重, 因为我们只需要embedding和attention
            # 而且我们一般可视化attention时, 也不会用到分类头的权重, 我们并不关注ESM内部关注什么, 在训练时关掉, 不要占据太大显存
            
            
        # 4. Extract Representations
        # token_repr: [Batch, Seq_Len + 2 (BOS/EOS + Padding), Dim] (shape of [batch_size of samples, seq_len including token+padding, representation_dim of ESM-2 = 1280])
        token_repr = results["representations"][self.esm_model.num_layers]
        
        # 5. Extract Attentions (⚠️ DeepPCT 'ResiduePairEmbeddings' to check, maybe useful)
        # attentions: [Batch, Num_Layers, Num_Heads, Seq_Len, Seq_Len]
        # In Separate-PPI mode, this is self-attention of the protein.
        # 上面我们提取的是每一层的attention map, 包含多个注意力头, 但是训练时爆显存,
        # 所以我们前面推理时设置need_head_weights=False, 那么这里就不会返回attention map, 返回None
        attentions = results.get("attentions", None)    
        
        # return the per-residue representations and attention maps
        # 正常情况下我们只需要token_repr, attentions一般不需要, 如果要返回最后一层? 也没必要
        return token_repr, attentions

    def extract_region_embeddings(self, 
                                token_reprs: torch.Tensor, 
                                regions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Description
        -----------
        Extract specific region embeddings from full sequence embeddings.
        从全序列Embedding中提取特定区域的Embedding, 即互作区域
        
        Args
        ----
            token_reprs: [Batch, Max_Len, Dim] - Full sequence embeddings (including BOS/EOS tokens)
            regions: List[(start, end)] - Start and End (both inclusive) indices for each sequence.
                     Note: indices should be 0-based relative to the original sequence.
                     ESM adds a BOS token at index 0, so real sequence starts at 1.
                     
        Returns
        -------
            region_embeddings: [Batch, Max_Region_Len, Dim] - Padded region embeddings 
            mask: [Batch, Max_Region_Len] - Mask for calculating mean/attention 
            
        Notes
        -----
        - 1, (start, end) in regions we interested are 0-based indices relative to the original sequence. 
        But ESM adds a BOS token at index 0, so the real sequence starts at index 1 in ESM embeddings.
        此处假设我们数据提取的(start, end)是相对于原始序列的0-based索引, 比如说CTCF (0,726); 但是ESM的token_reprs中, index 0是BOS token, 所以实际序列的token表示从index 1开始
        """
        # 输入数据token_reprs
        device = token_reprs.device   
        batch_size = len(regions)
        dim = token_reprs.shape[-1] 
        
        # 1. 调整索引以适配ESM的BOS token
        # ESM embedding: [BOS, Res1, Res2, ..., ResN, EOS]
        # for protein residue: User index 0 -> ESM index 1
        esm_regions = [(start + 1, end + 1) for start, end in regions]
        
        # 2. 找到最大区域长度用于Padding
        lengths = [end - start + 1 for start, end in esm_regions]
        max_len = max(lengths) if lengths else 0
        
        # 3. 提取并Pad
        # padded_regions是从全长序列tensor中提取的互作区域embedding, 并进行padding(统一成max_len)
        padded_regions = torch.zeros(batch_size, max_len, dim, device=device)
        # mask用于标记哪些是有效区域, 哪些是pad区域, Boolean tensor(真假地图: T为真实残基嵌入, F为填充)
        # 后续计算cross-attention时会用到mask, 避免pad区域对注意力计算的影响
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device) # 1 for valid, 0 for pad
        
        for i, (start, end) in enumerate(esm_regions):
            # 安全检查
            seq_len = token_reprs.shape[1]
            valid_end = min(end, seq_len - 2) # 避免越界，保留EOS前的内容
            valid_start = min(start, valid_end)
            
            length = valid_end - valid_start + 1
            if length > 0:
                # padded_regions中填充有效区域的embedding, 不需要BOS和EOS token
                # 因为原始的embedding全长已经是考虑了BOS/EOS的上下文
                # 且BOS/EOS的embedding通常不包含在互作区域内, 本来就是noise
                padded_regions[i, :length, :] = token_reprs[i, valid_start:valid_end+1, :]
                mask[i, :length] = True
                
        return padded_regions, mask   
    
    