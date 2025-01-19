import json
import os
from typing import List, Dict
from pathlib import Path
import numpy as np
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch, visit_num, projection_size]
        att_weights = self.second_linear(weights)  # [batch, visit_num, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch,num_classes, input_size],每个batchsize都有药物表征
        # t1 = self.third_linear.weight.mul(weighted_output)
        # t2 = t1.sum(dim=2)
        # t3 = t2.add(self.third_linear.bias)
        # logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
        drug_rep = weighted_output.mean(dim=0)

        # return (logits,drug_rep)
        return weighted_output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape query, key, and value
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Broadcast mask to (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)  # Fill masked positions with -1e9

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attended values
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Project attended values
        output = self.out_proj(attn_output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=384, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src, mask=None):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask=mask)
        return src


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        '''
            query: [batch_size, seq_len, hidden_dim]
            key: [batch_size, seq_len, hidden_dim]
            value: [batch_size, seq_len, hidden_dim]
        '''
        attn_output = self.multihead_attn(query, key, value, mask=mask)
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output


class TransformerCrossAttn(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, nhead=8, num_layers=2):
        super(TransformerCrossAttn, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.cross_attn_layers = nn.ModuleList([CrossAttention(d_model, nhead, dropout) for _ in range(num_layers)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        ) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x1, x2, mask=None):
        x1_pos = self.pos_encoder(x1)
        x2_pos = self.pos_encoder(x2)

        for i in range(len(self.cross_attn_layers)):
            # x1 attend to x2
            x1_pos = self.cross_attn_layers[i](query=x2_pos, key=x1_pos, value=x1_pos, mask=mask)
            x1_pos = x1_pos + self.feed_forward_layers[i](x1_pos)
            x1_pos = self.norm_layers[i](x1_pos)

            # x2 attend to x1
            x2_pos = self.cross_attn_layers[i](query=x1_pos, key=x2_pos, value=x2_pos, mask=mask)
            x2_pos = x2_pos + self.feed_forward_layers[i](x2_pos)
            x2_pos = self.norm_layers[i](x2_pos)

        return x1_pos, x2_pos

class DiffusionCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 定义Q,K,V投影矩阵
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 投影
        # [batch_size, hidden_dim] -> [batch_size, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        # [batch_size, num_heads, 1, head_dim] x [batch_size, num_heads, head_dim, 1]
        # -> [batch_size, num_heads, 1, 1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        # [batch_size, num_heads, 1, 1] x [batch_size, num_heads, 1, head_dim]
        # -> [batch_size, num_heads, 1, head_dim]
        output = torch.matmul(attn, v)
        
        # 重组多头注意力的结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 最终投影
        output = self.out_proj(output).squeeze(1)  # 移除序列维度
        
        return output
    
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, hidden_dim, device="cuda:0", time_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.device = device
        
        # 设置噪声调度
        self.beta = torch.linspace(beta_start, beta_end, time_steps).cpu()
        self.alpha = (1 - self.beta).cpu()
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).cpu()
        
        # 条件编码器 - 输出维度为hidden_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 下采样路径 - 保持hidden_dim维度
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.down3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 中间层的交叉注意力
        self.mid_cross_attn = DiffusionCrossAttention(hidden_dim)

        self.mid_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 上采样路径 - 考虑拼接后的维度
        self.up1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接后维度翻倍
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.up2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接后维度翻倍
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.up3 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 拼接后维度翻倍
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 输出层
        self.final = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer Norm 层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def get_time_embedding(self, t):
        """
        获取时间步的位置编码
        Args:
            t: 时间步
            d_model: 编码维度
        """
        device = t.device
        d_model = self.hidden_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float().to(device) / d_model))
        pos_enc_a = torch.sin(t.unsqueeze(1) * inv_freq)
        pos_enc_b = torch.cos(t.unsqueeze(1) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward_diffusion(self, x_0, t):
        """
        前向扩散过程
        Args:
            x_0: 原始表征
            t: 时间步
        """
        device = x_0.device
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar.to(device)[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar.to(device)[t])
        epsilon = torch.randn_like(x_0)
        
        x_t = sqrt_alpha_bar.unsqueeze(-1) * x_0 + \
              sqrt_one_minus_alpha_bar.unsqueeze(-1) * epsilon
        
        return x_t, epsilon

    def predicte_noise(self, x, condition, t):
        """
        使用UNet结构预测噪声
        Args:
            x: 输入表征 [batch_size, hidden_dim]
            condition: 条件表征 [batch_size, hidden_dim]
            t: 时间步 [batch_size]
        """
        # 获取时间编码
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # 编码条件信息
        condition_encoding = self.condition_encoder(condition)  # [batch_size, hidden_dim]
        
        # 下采样路径
        d1 = self.norm1(self.down1(x + t_emb))  # [batch_size, hidden_dim]
        d2 = self.norm2(self.down2(d1))         # [batch_size, hidden_dim]
        d3 = self.norm3(self.down3(d2))         # [batch_size, hidden_dim]
        
        # 中间层处理
        mid = self.mid_cross_attn(d3, condition_encoding, condition_encoding)
        mid = self.mid_block(mid)
        
        # 上采样路径
        u1 = self.up1(torch.cat([mid, d2], dim=-1))   # concat后 [batch_size, hidden_dim*2]
        u2 = self.up2(torch.cat([u1, d1], dim=-1))    # concat后 [batch_size, hidden_dim*2]
        u3 = self.up3(torch.cat([u2, x], dim=-1))     # concat后 [batch_size, hidden_dim*2]
        
        return self.final(u3)
    
    @torch.no_grad()
    def sample(self, condition):
        """
        基于条件生成表征
        Args:
            condition: 条件表征 [batch_size, condition_dim]
            device: 运算设备
        """
        device = condition.device
        n_samples = condition.shape[0]
        
        # 从标准正态分布采样 从随机噪声开始
        x = torch.randn(n_samples, self.hidden_dim).to(device)
        
        # 逐步去噪（受条件引导）
        for t in range(self.time_steps - 1, -1, -1):
            t_batch = torch.ones(n_samples, dtype=torch.long).to(device) * t
            
            # 预测噪声（加入条件） 
            predicted_noise = self.predicte_noise(x, condition, t_batch)
            
            alpha = self.alpha.to(device)[t]
            alpha_bar = self.alpha_bar.to(device)[t]
            beta = self.beta.to(device)[t]
            
            if t == 0:
                x = 1 / torch.sqrt(alpha) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)
            else:
                noise = torch.randn_like(x)
                x = 1 / torch.sqrt(alpha) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) + \
                    torch.sqrt(beta) * noise
                
        return x
        
    def forward(
            self,
            drugnet,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        # 使用药物网络获得学习的表征
        with torch.no_grad():
            x_0,condition = drugnet.get_patient_hist_rep(conditions,procedures,drugs)
        
        x_0 = x_0.to(self.device)
        condition = condition.to(self.device)
        
        # 随机采样时间步
        t = torch.randint(0, self.time_steps, (x_0.shape[0],), device=self.device)
        
        # 前向扩散 给原始表征添加噪声
        x_t, epsilon = self.forward_diffusion(x_0, t)
        
        # 预测噪声（包含条件） 模型学习预测噪声（基于条件信息）
        predicted_noise = self.predicte_noise(x_t, condition, t)
        
        # 计算损失 优化目标：使预测的噪声接近真实添加的噪声
        loss = F.mse_loss(predicted_noise, epsilon)

        return {
            "loss": loss,
        }

class LAMRec(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        super(LAMRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel"
        )
        # self.fusion_type = kwargs['fusion_type']
        self.beta = kwargs['beta']
        self.embedding_dim = embedding_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        # self.label_tokenizer = self.get_label_tokenizer(special_tokens=["<pad>"])
        self.label_tokenizer = self.get_label_tokenizer()
        self.label_num = self.label_tokenizer.get_vocabulary_size()

        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)
        # self.label_embeddings = nn.Embedding(
        #         self.label_tokenizer.get_vocabulary_size(),
        #         embedding_dim,
        #         padding_idx=self.label_tokenizer.get_padding_index()
        # )

        self.seq_encoder = TransformerCrossAttn(d_model=embedding_dim,dim_feedforward=embedding_dim)
        self.label_wise_attention = LabelAttention(embedding_dim, embedding_dim*2,self.label_num)
        self.patient_net = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc = nn.Linear(embedding_dim* 2,self.label_num)

        self.drugs_rep = nn.Parameter(torch.randn(self.label_num, embedding_dim))  # 可学习的药物表征
        self.drug_proj = nn.Linear(embedding_dim, embedding_dim)
        self.patient_proj = nn.Linear(embedding_dim, embedding_dim)

        self.ddi_adj = self.generate_ddi_adj().cpu()
        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj)



    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def reshape_and_mask_tensor(self,input_tensor, mask):
            '''
            mask：有效的visit
            输入的input_tensor是(batchsize,visit_num,hidden_dim)
            masked_tensor(batchsize*visit_num,hidden_dim)
            '''
            # 确保输入张量和掩码的维度正确
            assert input_tensor.shape[:2] == mask.shape, "输入张量和掩码的前两个维度必须相同"
            
            # 将掩码扩展到与输入张量相同的维度
            expanded_mask = mask.unsqueeze(-1).expand_as(input_tensor)
            
            # 使用掩码选择有效的访问
            masked_tensor = input_tensor[expanded_mask].view(-1, input_tensor.shape[-1])
            
            return masked_tensor
    
    def restore_tensor_shape(self, masked_tensor, mask):
        """
        将 masked_tensor 恢复为原始的三维形状。
        
        参数:
        masked_tensor: 形状为 (batchsize*visit_num, hidden_dim) 的张量
        mask: 形状为 (batchsize, visit_num) 的布尔掩码
        hidden_dim: 隐藏维度的大小
        
        返回:
        restored_tensor: 形状为 (batchsize, visit_num, hidden_dim) 的张量
        """
        batchsize, visit_num = mask.shape
        
        # 创建一个全零张量作为结果
        hidden_dim = masked_tensor.shape[-1]
        restored_tensor = torch.zeros(batchsize, visit_num, hidden_dim, device=masked_tensor.device)
        
        # 使用 mask 来填充 restored_tensor
        valid_visits = mask.sum()
        assert valid_visits == masked_tensor.shape[0], "masked_tensor 的样本数与 mask 中的有效访问数不匹配"
        
        masked_tensor_idx = 0
        for i in range(batchsize):
            for j in range(visit_num):
                if mask[i, j]:
                    restored_tensor[i, j] = masked_tensor[masked_tensor_idx]
                    masked_tensor_idx += 1
        
        return restored_tensor

    def forward(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)

        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)

        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        conditions = self.embeddings['conditions'](conditions)
        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        procedures = self.embeddings['procedures'](procedures)
        procedures = torch.sum(procedures, dim=2)

        loss = 0
        
        diag_out, proc_out = self.seq_encoder(conditions, procedures, mask)

        patient_hist_rep = torch.cat((diag_out, proc_out), dim=-1)
        patient_hist_rep = self.patient_net(patient_hist_rep)
        patient_cur_rep = patient_hist_rep[:, -1, :]

        def caculate_loss(patient_cur_rep):
            logits = torch.mm(patient_cur_rep, self.drugs_rep.T) # (batch, label_num)
            bce_loss = F.binary_cross_entropy_with_logits(logits, curr_drugs)
            return logits,bce_loss
        
        logits,bce_loss = caculate_loss(patient_cur_rep)
        loss += bce_loss
        
        # 计算ddi
        y_prob = torch.sigmoid(logits)
        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        current_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())
        if current_ddi_rate >= 0.06:
            mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
            batch_ddi_loss = (
                torch.sum(mul_pred_prob.mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2
            )
            loss += 0.1 * batch_ddi_loss

        # 使用diffsuion模型
        if 'diffusion_model' in kwargs:
            sum_drugs_rep = torch.mm(curr_drugs,self.drugs_rep)
            valid_counts = curr_drugs.sum(dim=1, keepdim=True)  # 计算每个样本中有效药物的数量
            sum_drugs_rep = sum_drugs_rep / valid_counts # 归一化
            generated_patient_rep = kwargs['diffusion_model'].sample(sum_drugs_rep)
            logits,bce_loss = caculate_loss(generated_patient_rep)
            loss += self.beta*bce_loss

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs
        }
    
    def get_patient_hist_rep(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
    ):
        # 准备标签
        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)

        # 编码diagnose和procedure
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        conditions = self.embeddings['conditions'](conditions)
        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        procedures = self.embeddings['procedures'](procedures)
        procedures = torch.sum(procedures, dim=2)
        
        diag_out, proc_out = self.seq_encoder(conditions, procedures, mask)

        patient_hist_rep = torch.cat((diag_out, proc_out), dim=-1)
        patient_hist_rep = self.patient_net(patient_hist_rep)
        
        sum_drugs_rep = torch.mm(curr_drugs,self.drugs_rep)
        valid_counts = curr_drugs.sum(dim=1, keepdim=True)  # 计算每个样本中有效药物的数量
        sum_drugs_rep = sum_drugs_rep / valid_counts  # 归一化
        return patient_hist_rep[:,-1,:],sum_drugs_rep



    
