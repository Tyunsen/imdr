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
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, projection_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1,
                                                                                2)  # [batch_size,num_classes, seq_len]
        weighted_output = att_weights @ x  # [batch_size,num_classes, input_size]
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

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
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
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


class MultiViewContrastiveLoss(nn.Module):
    def __init__(self, temperature=8):
        super(MultiViewContrastiveLoss, self).__init__()
        self.temperature = temperature

    def compute_joint(self, x_out, x_tf_out):
        # produces variable that requires grad (since args require grad)

        bn, k = x_out.size()
        assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        # p_i_j = torch.nansum(p_i_j, dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def forward(self, x_out, x_tf_out, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        if len(x_out.size()) == 3:
            x_out = x_out.mean(dim=1)  # (batch_size, hidden_dim)
            x_tf_out = x_tf_out.mean(dim=1)  # (batch_size, hidden_dim)

        x_out, x_tf_out = F.softmax(x_out, dim=-1), F.softmax(x_tf_out, dim=-1)
        _, k = x_out.size()

        p_i_j = self.compute_joint(x_out, x_tf_out)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss = - p_i_j * (torch.log(p_i_j) \
                          - self.temperature * torch.log(p_j) \
                          - self.temperature * torch.log(p_i))

        loss = loss.sum()

        return loss


class LAMRec(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        device = kwargs.get('device', torch.device("cpu"))
        super(LAMRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel"
        )

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()

        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        self.alpha = 0.1
        self.beta = 0.1
        self.seq_encoder = TransformerCrossAttn(d_model=embedding_dim,dim_feedforward=embedding_dim)
        self.multi_view_cl = MultiViewContrastiveLoss()
        self.label_wise_attention = LabelAttention(embedding_dim * 2, embedding_dim,
                                                   self.label_tokenizer.get_vocabulary_size())

        # self.ddi_adj = self.generate_ddi_adj().to(self.device)
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


    def forward(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)

        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)
        # conditions = self.conditions_embeddings(conditions)
        conditions = self.embeddings['conditions'](conditions)
        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)
        # procedures = self.procedures_embeddings(procedures)
        procedures = self.embeddings['procedures'](procedures)
        procedures = torch.sum(procedures, dim=2)

        diag_out, proc_out = self.seq_encoder(conditions, procedures, mask)

        mvcl = self.multi_view_cl(diag_out, proc_out)

        patient_rep = torch.cat((diag_out, proc_out), dim=-1)

        logits = self.label_wise_attention(patient_rep)

        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)

        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        loss = F.binary_cross_entropy_with_logits(logits, curr_drugs)

        
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
            loss += self.alpha * batch_ddi_loss

        return {
            "loss": loss + self.beta * mvcl,
            "y_prob": y_prob,
            "y_true": curr_drugs,
            # "loss_bce_batch":F.binary_cross_entropy_with_logits(logits, curr_drugs,reduction='none'),
            # "loss_bce":F.binary_cross_entropy_with_logits(logits, curr_drugs),
            "entropy_batch":F.binary_cross_entropy(y_prob, y_prob,reduction='none').mean(dim=-1) ,
        }


def shownan(input,name):
    nan_mask = torch.isnan(input)
    nan_count = nan_mask.sum().item()
    nan_positions = torch.nonzero(nan_mask)

    print("---"*30)
    print(f"{name} 中 NaN 的数量: {nan_count}")
    print(f"{name} 的总元素数量: {input.numel()}")
    print(f"{name} NaN 占比: {nan_count / input.numel() * 100:.2f}%")

    if nan_count > 0:
        print("{name} NaN 的前几个位置:")
        print(nan_positions[:5])  # 只打印前5个位置作为示例
    print("---"*30)