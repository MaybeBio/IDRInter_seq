import torch
import torch.nn as nn
import torch.nn.functional as F # 神经网络操作函数库，无自带可学习参数
import lightning as L # pip install lightning
from sequence_embedding import SeqEmbedModel # ESM-2 embedding model
from typing import *
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall # 从torchmetrics导入计算F1, Accuracy, AUROC的工具, 用于多维度评估模型性能

class InterProtein_DecoderLayer(nn.Module):
    """
    Description
    ----------
    Decoder block for inter-protein interactions,
    we extract intra-protein features using ESM-2 embedding model(self-attention), then use this module to model inter-protein interactions(cross-attention).
    输入特征 → 注意力层 → norm1 → FFN → norm2 → 输出到下一个Decoder块
        
    Custom Transformer Decoder Layer for Inter-Protein Attention.
    Standard Transformer Decoder usually has Self-Attention -> Cross-Attention -> FeedForward.
    Here we simplify it for "Inter-Protein" modeling as: Cross-Attention -> FeedForward.
    Because for a specific protein region, we care more about how it attends to its partner protein,
    rather than its internal self-attention (which ESM-2 encoder has already captured well).
    
    Architecture:
    Input Query (Protein A Region) ->
    [ Cross-Attention (Query=A, Key/Value=Protein B Region) ] ->
    [ Add & Norm ] ->
    [ Feed Forward Network ] ->
    [ Add & Norm ] ->
    Output (Updated Protein A Features)

    Notes
    ------
    - 1, this module contains attention layer(used as cross-attention) + feed-forward layer, a non-standard transformer decoder block(without masking attention)
    - 2, 整体数据流向是: 输入特征 → 注意力层 → norm1 → FFN → norm2 → 输出到下一个Decoder块,
    所以我们在做FFN时还是要确保维度不变, 以便后续堆叠多个Decoder块(也就是说先升维再降维, 最后维度还是要确保和最开始ESM-2 embedding输入维度一致)
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """    
        Description
        -----------
        initialize inter-protein decoder block, including attention layer and feed-forward layer
        
        Args
        ----
            embed_dim: embedding dimension of input features
            num_heads: number of attention heads, should divide embed_dim evenly
            dropout: dropout rate
        
        Notes
        -----
        - 1, Multi-Head Attention 的底层计算要求 每个注意力头的维度 d_k 相等，而 d_k = d_model / nhead, 总之embed_dim必须要能被num_heads整除;
        本质是将 "单一注意力计算" 拆分为多个并行的 "子注意力头", 每个头专注于输入特征的不同维度, 最终融合结果以捕捉更全面的依赖关系, 
        所以head数不能太少也不能太多
        """
        
        super().__init__()
        
        # multi-head self-attention layer
        # set batch_first=True here to make input shape (batch_size, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        
        # Layer normalization for attention output
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # Dropout layer for residual connections
        self.dropout = nn.Dropout(dropout)
        
        # feed-forward layer
        # 增加非线性变换能力, 整合Attention提取的特征
        # a 2-layer MLP with ReLU activation + dropout
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # 升维1280->5120
            nn.GELU(),                             # 非线性激活, GELU效果优于ReLU
            nn.Dropout(dropout),                 # 防止过拟合, Dropout层一般放在激活函数后面
            nn.Linear(embed_dim * 4, embed_dim), # 降维5120-> 1280,恢复原始维度
        )
        
        # layer normalization for feed-forward output
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor = None, need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """    
        Description
        -----------
        forward process of inter-protein decoder block
        
        Args
        ----
            query(protein A): input features as query from protein A, shape (batch_size, seq_len_q, embed_dim)
            key(protein B): input features as key from protein B, shape (batch_size, seq_len_k, embed_dim)
            value(protein B): input features as value from protein B, shape (batch_size, seq_len_k, embed_dim)
            key_padding_mask: attention mask, shape (batch_size, seq_len_k), used to mask padding tokens in key
            need_weights: whether to return attention weights
        
        Returns
        -------
            x: output features, after inter-protein decoder block, shape (batch_size, seq_len_k, embed_dim)
            attn_weight: attention weights if need_weights=True, else None
        """
        
        # 1, Attention layer
        # multi-head attention
        # Note: here we use query from protein A, key and value from protein B to model inter-protein interactions
        # so this is not self-attention but cross-attention
        # 本蛋白的特征query(希望被更新), 看着对方蛋白的特征(key/value提供上下文信息)更新自己
        attn_output, attn_weight = self.attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)  
        
        # Dropout + Residual Connection + Norm (Add & Norm)
        # post-norm: Dropout 正则化 → 残差连接 → 层归一化
        # 结构: x = Norm(x + Dropout(SubLayer(x))) --> Post-Norm 结构
        x = self.attn_norm(query + self.dropout(attn_output))
        
        # 2, Feed-forward layer
        ffn_output = self.ffn(x)
        
        # Dropout + Residual Connection + Norm (Add & Norm)
        # post-norm: Dropout 正则化 → 残差连接 → 层归一化
        x = self.ffn_norm(x + self.dropout(ffn_output))
        
        return x, attn_weight
    
    
class InterProtein_Decoder(nn.Module):
    """
    Description
    ----------
    Stack multiple inter-protein decoder blocks to model inter-protein interactions,
    Interaction Module simulating Transformer Decoder but without masked self-attention.
    Captures interaction between Protein A (query) and Protein B (key/value).
    通过堆叠多个Decoder块来增强模型对复杂交互关系的建模能力
    结构: 输入特征 → Decoder Block 1 → Decoder Block 2 → ...
    交互模块: 模拟了Transformer Decoder的行为, 但去掉了生成任务用的Masked Self-Attention
    
    """
    
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """    
        Description
        -----------
        initialize inter-protein decoder, stack multiple inter-protein decoder blocks
        
        Args
        ----
            num_layers: number of decoder blocks to stack
            embed_dim: embedding dimension of input features
            num_heads: number of attention heads
            dropout: dropout rate
        
        """
        
        super().__init__()
        
        # stack multiple inter-protein decoder blocks
        self.layers = nn.ModuleList([
            InterProtein_DecoderLayer(embed_dim=embed_dim,
                                      num_heads=num_heads,
                                      dropout=dropout)
            for _ in range(num_layers)
        ])
        
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor = None, return_attn_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """    
        Description
        -----------
        forward process of inter-protein decoder
        
        Args
        ----
            query(protein A): input features as query from protein A, shape (batch_size, seq_len_q, embed_dim)
            key(protein B): input features as key from protein B, shape (batch_size, seq_len_k, embed_dim)
            value(protein B): input features as value from protein B, shape (batch_size, seq_len_k, embed_dim)
            key_padding_mask: attention mask, shape (batch_size, seq_len_k), used to mask padding tokens in key
            return_attn_weights: whether to return attention weights from all layers
        
        Returns
        -------
            output features after inter-protein decoder, shape (batch_size, seq_len_k, embed_dim)
        
        Notes
        -----
        - 1, ⚠️ 训练期间我们一般不需要返回attention weights, 只有在可视化时才需要, 所以默认False;
        就是attention map是可解释性接口, 训练时不需要返回, 只在推理可视化时返回
        """
        
        output = query
        attn_weights = []
        
        # pass through each decoder block
        for layer in self.layers:
            # 逐层传递: Layer1的输出作为Layer2的输入(即Query不断更新)
            # 注意: key和value(即对方蛋白的特征)在每一层都保持原样, 没有更新
            output, attn_weight = layer(query=output, key=key, value=value, key_padding_mask=key_padding_mask, need_weights=return_attn_weights)
            if return_attn_weights:
                attn_weights.append(attn_weight)
            
        if return_attn_weights:
            return output, attn_weights
        return output
    
    
class IDRinter_Seq(L.LightningModule): 
    """   
    Description
    -----------
    Frozen ESM-2 (33 layer Encoder) -> Inter-Protein Decoder (2 layer Decoder, non-standard Transformer Decoder without masking)
    
    Architecture / 架构流程:
    1. Encoder: ESM-2 (Frozen) -> 理解单体全长序列
    2. Region Extraction -> 聚焦互作区域
    3. Decoder: IDRDecoder (Cross-Attention) -> 理解相互作用 (A关注B, B关注A)
    4. Classifier -> 综合判别
    
    Notes
    -------
    - 1, 用于处理蛋白质序列的模块, 包含ESM-2预训练模型和Inter-Protein Decoder模块;
    注意我们此脚本中只定义了Decoder部分, 因为我们只需要用到Decoder来建模蛋白质间的交互关系;
    而encoder部分我们直接使用ESM-2预训练模型来提取蛋白质的序列特征, 而且ESM-2本身是多个Transformer Encoder块堆叠而成的,
    所以我们此处就不需要再额外定义Encoder模块了.
    """   
    
    def __init__(self, esm_model_path: str = None, num_decoder_layers: int = 2, embed_dim: int = 1280, num_heads: int = 8, dropout: float = 0.1, learning_rate: float = 1e-4):
        """ 
        Description
        ------------
        Initialize IDRinter sequence module, including ESM-2 model and inter-protein decoder
        
        Args
        -----
            esm_model_path: pre-trained ESM-2 model path for sequence embedding extraction
            num_decoder_layers: number of inter-protein decoder layers
            embed_dim: embedding dimension from ESM-2 model
            num_heads: number of attention heads in inter-protein decoder
            dropout: dropout rate for inter-protein decoder
            learning_rate: learning rate for optimizer
        
        """
        
        super().__init__()
        # save hyperparameters for LightningModule, including each argument in __init__
        # so we can check them later via self.hparams and in saved checkpoints
        self.save_hyperparameters()
        
        # 1. Sequence Embedding Model: ESM-2 (Frozen)
        self.embedding_model = SeqEmbedModel(esm_model_path=esm_model_path)
        
        # Freeze ESM-2 parameters
        # 我们不微调 ESM-2，因为它太大且已经训练得很好了, 我们只把它当成一个"特征提取器";
        # 设置反向传播时不会更新这些层的权重，大大节省显存和计算量
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
        # get embedding dimension from ESM-2 model
        # 1280 here for ESM-2 650M
        embed_dim = self.embedding_model.get_embedding_dim()    

        # 2. Inter-Protein Decoder
        # 交互解码器, 用于建模蛋白质间的交互关系
        # 此处我们定义了两个解码器, 分别处理A对B的关注和B对A的关注
        # decoder_a2b: Protein A 关注 Protein B, A看着B思考, 让A的特征依据B的特征进行更新
        self.decoder_a2b = InterProtein_Decoder(num_layers=num_decoder_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # decoder_b2a: Protein B 关注 Protein A, B看着A思考, 让B的特征依据A的特征进行更新
        self.decoder_b2a = InterProtein_Decoder(num_layers=num_decoder_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 3. Classifier Head
        # 分类器头, 综合解码器输出进行最终判别, 本质是1个MLP
        # 输入维度: embed * 4, 因为我们后面会拼接4个向量:
        # (1) decoder_a2b的输出, (2) decoder_b2a的输出, (3) ESM-2的A特征, (4) ESM-2的B特征
        # 线性层-(归一化层->激活函数层->Dropout层)-线性层
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 1024), # 降维, 第1层(1280*4)映射到1024维度
            nn.BatchNorm1d(1024),         # 批归一化, 加速收敛, 防止梯度消失/爆炸
            nn.GELU(),                             # 非线性激活, GELU效果优于ReLU
            nn.Dropout(0.3),                 # 防止过拟合, decoder中默认随机丢弃0.1, 此处可以稍微大一点, 0.3左右 
            nn.Linear(1024, 256),       # 降维, 第2层1024映射到256维度
            nn.BatchNorm1d(256),         # 批归一化
            nn.GELU(),                             # 非线性激活
            nn.Dropout(0.3),                 # 防止过拟合
            nn.Linear(256, 1)               # 最终映射到1维, 用于二分类任务的logit输出
        )
        
        
        # loss function
        # 二分类交叉熵损失函数, 适用于二分类任务
        # 参考: https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
        # pos_weight = num_negatives / num_positives, 可以在训练时动态调整正负样本权重, 以应对类别不平衡问题
        # ⚠️ 我们的训练数据是2:1的正负样本比例, 所以可以设置 pos_weight=2 来让模型更关注正样本
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
        # 但是加了之后发现效果变差了, 可能是因为2:1还不算样本不平衡, 先不加了
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics: Initialize commonly use classification metrics
        # 初始化分类任务常用指标: 准确率, F1, AUROC, 精确率, 召回率
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")        
    # in lightning, forward defines the prediction/inference actions    
    def forward(self, seqs_a: List[str], seqs_b: List[str], regions_a: List[Tuple[int, int]], regions_b: List[Tuple[int, int]], return_attn: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        """    
        Description
        -----------
        forward process of IDRinter sequence module
        
        Args
        ----
            seqs_a: list of protein A sequences in the batch
            seqs_b: list of protein B sequences in the batch
            regions_a: list of tuples indicating interaction regions in protein A, each tuple is (start_idx, end_idx)
            regions_b: list of tuples indicating interaction regions in protein B, each tuple is (start_idx, end_idx)
            return_attn: whether to return attention weights for visualization
        
        Returns
        -------
            logits: output logits from classifier head, shape (batch_size, 1)
        """
        
        
        #----------------------#
        # 下面的计算共存embed_a和embed_b 两个全长序列张量, 这意味着在计算B时embed_a还在显存中, 非常占据显存
        # 应该串行处理: 算出A->提取A区域->立即删除A全长embed, 然后再算B->提取B区域->删除B全长embed
        '''
        # 1. ESM-2 Embedding Extraction
        # extract sequence embeddings for protein A and B using frozen ESM-2 model
        # get embeddings: shape (batch_size, seq_len, embed_dim)
        # 因为ESM-2只是前向推理, 不需要计算梯度, 防止爆显存
        with torch.no_grad():
            embed_a, _ = self.embedding_model.generate_protein_embeddings(seqs_a) # Protein A embeddings
            embed_b, _ = self.embedding_model.generate_protein_embeddings(seqs_b)  # Protein B embeddings
        
        # 2. Region Extraction
        # extract interaction regions from embeddings based on provided regions
        # get region embeddings: shape (batch_size, region_len, embed_dim)
        # 因为每个样本截出来的长度不一致, 此处返回的是padding到最大长度的统一结果
        # region_embed: padded region embeddings, shape (batch_size, max_region_len, embed_dim)
        # mask: mask for calculating mean/attention, shape (batch_size, max_region_len), 1表示有效残基位, 0表示padding位
        region_embed_a, mask_a = self.embedding_model.extract_region_embeddings(embed_a, regions_a)  # Protein A region embeddings
        region_embed_b, mask_b = self.embedding_model.extract_region_embeddings(embed_b, regions_b)  # Protein B region embeddings
        '''
        #----------------------#
        
        # --- Process Protein A ---
        with torch.no_grad():
            # 开启混合精度，大幅减少 Attention Map 的显存占用
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embed_a, _ = self.embedding_model.generate_protein_embeddings(seqs_a) 

        # 提取区域特征 (Region Embeddings 通常比 全长 Embeddings 小得多)
        region_embed_a, mask_a = self.embedding_model.extract_region_embeddings(embed_a, regions_a)
        
        # ⚠️ 既然已经拿到了 region_embed_a，巨大的全长 embed_a 就不再需要了，立即删除释放显存
        del embed_a 
        
        # --- Process Protein B ---
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embed_b, _ = self.embedding_model.generate_protein_embeddings(seqs_b)
        
        # 提取区域特征
        region_embed_b, mask_b = self.embedding_model.extract_region_embeddings(embed_b, regions_b)
        
        # ⚠️ 同样立即删除全长 embed_b
        del embed_b
        
        #----------------------#
        
        
        # invert mask for key_padding_mask (True for padding positions)
        # ⚠️ 注意到attention层key_padding_mask的定义是True表示padding位, False表示有效位, 所以此处需要反转一下
        pad_mask_a = ~mask_a  
        pad_mask_b = ~mask_b  
        
        # 3. Inter-Protein Decoder
        # Protein A attends to Protein B
        if return_attn:
            # A is query, B is key/value, 包含了B信息的A特征
            inter_a, attn_a2b = self.decoder_a2b(query=region_embed_a, key=region_embed_b, value=region_embed_b, key_padding_mask=pad_mask_b, return_attn_weights=True)
            # B is query, A is key/value, 包含了A信息的B特征
            inter_b, attn_b2a = self.decoder_b2a(query=region_embed_b, key=region_embed_a, value=region_embed_a, key_padding_mask=pad_mask_a, return_attn_weights=True)
        else:
            inter_a = self.decoder_a2b(query=region_embed_a, key=region_embed_b, value=region_embed_b, key_padding_mask=pad_mask_b)
            inter_b = self.decoder_b2a(query=region_embed_b, key=region_embed_a, value=region_embed_a, key_padding_mask=pad_mask_a)
        
        # 4. Pooling
        # Pooling: mean pooling over sequence length dimension to get fixed-size representation
        # 将变长的序列特征通过均值池化转换为固定长度的向量表示， 从[Batch, Max_Region_Len, embed_Dim]->[Batch, embed_Dim]
        def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """    
            Description
            -----------
            Calculate masked mean pooling over sequence length dimension
            将变长序列通过掩码进行均值池化, 得到固定长度向量表示, 每一条序列将所有残基的embedding取平均, 忽略padding部分
            
            Args
            ----
                tensor: input tensor of shape (batch_size, seq_len, embed_dim)
                mask: boolean mask of shape (batch_size, seq_len), True for valid positions, False for padding positions
            
            Returns
            -------
                pooled: mean pooled tensor of shape (batch_size, embed_dim)
            """
            # expand mask to match tensor shape
            # 显示转float
            mask_float = mask.unsqueeze(-1).float()  # shape (batch_size, seq_len, 1)
            # apply mask to tensor
            # 相乘掩码过滤+求和
            masked_tensor = tensor * mask_float
            # sum over sequence length dimension
            sum_tensor = masked_tensor.sum(dim=1)  # shape (batch_size, embed_dim)
            # count valid positions for each sample, avoid division by zero
            valid_counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)   # shape (batch_size, 1)
            # calculate mean
            pooled = sum_tensor / valid_counts  # shape (batch_size, embed_dim)
            return pooled
        
        # 将region_embed_a, region_embed_b, inter_a, inter_b分别进行masked mean pooling, 获取有效残基的均值表示
        # shape (batch_size, embed_dim)
        pool_a = masked_mean(region_embed_a, mask_a) # A初始特征均值表示
        pool_b = masked_mean(region_embed_b, mask_b) # B初始特征均值表示
        pool_inter_a = masked_mean(inter_a, mask_a) # A关注B后的特征均值表示
        pool_inter_b = masked_mean(inter_b, mask_b) # B关注A后的特征均值表示
        
        # 4. Classifier Head         
        # Concatenate all pooled features
        # 拼接4个向量, 维度变为了 1280 (A) + 1280 (B) + 1280 (InterA) + 1280 (InterB) = 5120
        # 列1列, 沿着列的维度对每一行进行拼接
        combined = torch.cat([pool_a, pool_b, pool_inter_a, pool_inter_b], dim=1)  # shape (batch_size, embed_dim * 4)
        
        # Get logits from classifier head
        logits = self.classifier(combined)  # shape (batch_size, 1)
        
        if return_attn:
            return logits, {"a2b": attn_a2b, "b2a": attn_b2a}
        return logits

    
    # training_step defines the train loop. It is independent of forward
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """    
        Description
        -----------
        training step for IDRinter sequence module
        
        Args
        ----
            batch: input batch containing sequences and labels, tuple of (seqs_a, seqs_b, regions_a, regions_b, labels)
            batch_idx: index of the current batch
        
        Returns
        -------
            loss: computed loss for the batch
            accuracy: computed accuracy for the batch
        """
        
        # 1. batch unpacking 
        # batch 是从 DataLoader 里出来的，参考 src/data_loader.py 中的 collate_fn函数
        seqs_a, seqs_b, regions_a, regions_b, labels = batch
        
        # 2. forward pass
        logits = self.forward(seqs_a, seqs_b, regions_a, regions_b) # or just self(seqs_a, seqs_b, regions_a, regions_b)
        
        # 3. compute loss and accuracy
        # binary cross-entropy loss, squeeze logits from (batch_size, 1) to (batch_size,) to match labels shape
        loss = self.criterion(logits.squeeze(1), labels.float())
        # squeeze logits and compute the sigmoid to get probabilities, then threshold at 0.5 for binary predictions
        # ⚠️ 注意 logits 需要先经过 sigmoid 转换为概率值, 然后再与标签比较, 这里其实可以直接得到1个概率值, 所以我们后续的任务其实不一定要输出1个label, 也可以预测输出1个概率
        # Calculate metrics
        probs = torch.sigmoid(logits.squeeze(1))
        self.train_acc(probs, labels.int())
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(labels))
        
        # 5. return loss for backpropagation
        # lightning will handle optimizer step and backward pass, same as loss.backward() and optimizer.step()
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """    
        Description
        -----------
        validation step for IDRinter sequence module
        
        Args
        ----
            batch: input batch containing sequences and labels, tuple of (seqs_a, seqs_b, regions_a, regions_b, labels)
            batch_idx: index of the current batch
        """
        
        # 1. batch unpacking 
        seqs_a, seqs_b, regions_a, regions_b, labels = batch
        
        # 2. forward pass
        logits = self.forward(seqs_a, seqs_b, regions_a, regions_b)
        
        # 3. compute loss and accuracy
        loss = self.criterion(logits.squeeze(1), labels.float())
        
        
        # Calculate metrics
        probs = torch.sigmoid(logits.squeeze(1))
        # Cast labels to int for metrics that require integer targets
        labels_int = labels.int()
        
        self.val_acc(probs, labels_int)
        self.val_f1(probs, labels_int)
        self.val_auroc(probs, labels_int)
        self.val_precision(probs, labels_int)
        self.val_recall(probs, labels_int)
        
        # 4. log validation loss and accuracy
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True, batch_size=len(labels))
        self.log('val_f1', self.val_f1, on_epoch=True, batch_size=len(labels))
        self.log('val_auroc', self.val_auroc, on_epoch=True, batch_size=len(labels))
        self.log('val_precision', self.val_precision, on_epoch=True, batch_size=len(labels))
        self.log('val_recall', self.val_recall, on_epoch=True, batch_size=len(labels))
        
        return loss

    def test_step(self, batch, batch_idx):
        """
        Description
        -----------
        test step for IDRinter sequence module

        Args
        ----
            batch: input batch containing sequences and labels, tuple of (seqs_a, seqs_b, regions_a, regions_b, labels)
            batch_idx: index of the current batch 
            
        """
        seqs_a, seqs_b, regions_a, regions_b, labels = batch
        logits = self(seqs_a, seqs_b, regions_a, regions_b)
        loss = self.criterion(logits.squeeze(1), labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits.squeeze(1))
        labels_int = labels.int()
        
        self.test_acc(probs, labels_int)
        self.test_f1(probs, labels_int)
        self.test_auroc(probs, labels_int)
        self.test_precision(probs, labels_int)
        self.test_recall(probs, labels_int)

        self.log('test_loss', loss, on_epoch=True, batch_size=len(labels))
        self.log('test_acc', self.test_acc, on_epoch=True, batch_size=len(labels))
        self.log('test_f1', self.test_f1, on_epoch=True, batch_size=len(labels))
        self.log('test_auroc', self.test_auroc, on_epoch=True, batch_size=len(labels))
        self.log('test_precision', self.test_precision, on_epoch=True, batch_size=len(labels))
        self.log('test_recall', self.test_recall, on_epoch=True, batch_size=len(labels))
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """    
        Description
        -----------
        configure optimizer for training
        
        Returns
        -------
            optimizer: configured optimizer
        """
        
        # use AdamW optimizer for training (AdamW优化器, 权重衰减比Adam更合理)
        # lr 从 self.hparams.learning_rate 读取 (即 __init__ 中保存的)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
            
        # 学习率调度器：ReduceLROnPlateau
        # 作用：当验证集 loss ('val_loss') 不再下降时，自动把学习率减半。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',  # 监测指标越小越好 (loss)
                factor=0.5,  # 每次把 LR 乘以 0.5
                patience=5   # 忍耐 5 个 epoch 不下降再调整
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" # val_loss为监测指标
            }
        }