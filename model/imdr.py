from collections import defaultdict
import os
import sys
import pickle
from typing import List, Dict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model.layers import GCN,TransformerSelfAttn,TransformerCrossAttn
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.tokenizer import Tokenizer
     
class IMDR(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        super(IMDR, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel"
        )
        self.alpha = kwargs.get('alpha', 0)
        self.beta = kwargs.get('beta', 0)
        self.gamma = kwargs.get('gamma', 0)
        self.gat_num_heads = kwargs.get('gat_num_heads', 4)

        self.difficulty_weight = nn.Parameter(torch.zeros(3))

        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        diag_tokens=self.dataset.get_all_tokens(key="conditions")
        proc_tokens=self.dataset.get_all_tokens(key="procedures")
        self.all_tokens = diag_tokens+proc_tokens

        self.feat_tokenizers = Tokenizer(
            tokens=self.all_tokens,
            special_tokens=["<pad>", "<unk>"],
        )

        self.embedding_dim = embedding_dim
        self.label_num = self.get_label_tokenizer().get_vocabulary_size()

        self.label_tokenizer = self.get_label_tokenizer()

        self.embeddings = nn.Embedding(
            self.feat_tokenizers.get_vocabulary_size(),
            embedding_dim,
            padding_idx=self.feat_tokenizers.get_padding_index(),
        )

        self.seq_encoder = TransformerSelfAttn(d_model=embedding_dim,dim_feedforward=embedding_dim,num_layers=1)
        self.cross_attention = TransformerCrossAttn(
            d_model=embedding_dim, 
            dim_feedforward=embedding_dim,
            num_layers=1
        )

        self.patient_net = nn.Linear(embedding_dim*2, embedding_dim)

        self.ddi_adj = self.generate_ddi_adj().cpu()
        
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj)

        self.fc = nn.Linear(embedding_dim,self.label_num)

        if os.path.exists(os.path.join(BASE_CACHE_PATH, "ehr_adj.npy")):
            self.ehr_adj = torch.tensor(np.load(os.path.join(BASE_CACHE_PATH, "ehr_adj.npy")))
        else:
            self.ehr_adj = self.generate_ehr_adj()
        self.ehr_gcn = GCN(adj=self.ehr_adj, hidden_size=embedding_dim, dropout=0.1)
        self.ddi_gcn = GCN(adj=self.ddi_adj, hidden_size=embedding_dim, dropout=0.1)
        self.x = nn.Parameter(torch.FloatTensor(1))
        

    def generate_ehr_adj(self) -> torch.tensor:
        """Generates the EHR graph adjacency matrix."""
        label_size = self.label_tokenizer.get_vocabulary_size()
        ehr_adj = torch.zeros((label_size, label_size))
        for sample in self.dataset:
            curr_drugs = sample["drugs"]
            encoded_drugs = self.label_tokenizer.convert_tokens_to_indices(curr_drugs)
            for idx1, med1 in enumerate(encoded_drugs):
                for idx2, med2 in enumerate(encoded_drugs):
                    if idx1 >= idx2:
                        continue
                    ehr_adj[med1, med2] = 1
                    ehr_adj[med2, med1] = 1
        return ehr_adj
    
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
    
    def get_dynamic_weights(self, 
            mask: List[int],
            eps: float = 1e-12,
            **kwargs
        ) -> torch.Tensor:
        """计算每个样本的动态权重
        
        Args:
            difficulty: 形状为 [batch_size] 的难度张量
            
        Returns:
            形状为 [batch_size] 的权重张量
        """
        tokens_complexity = [kwargs['complexity'][i] for i in mask]
        tokens_complexity = torch.tensor(tokens_complexity,dtype=torch.float32,device=self.device)
        tokens_complexity_scores = (tokens_complexity - tokens_complexity.min()) / (tokens_complexity.max() - tokens_complexity.min() + eps)

        ddi_rate = [kwargs['ddi_rate'][i] for i in mask]
        ddi_scores = torch.tensor(ddi_rate,dtype=torch.float32,device=self.device)

        view_score = [kwargs['view_score'][i] for i in mask]
        view_scores = torch.tensor(view_score,dtype=torch.float32,device=self.device)

        scores = torch.stack([view_scores, ddi_scores, tokens_complexity_scores], dim=1)  # [batch, 3]
        difficulty_weight = F.softmax(self.difficulty_weight, dim=0)
        difficulty = torch.matmul(difficulty_weight, scores.T)  # [batch]
        difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + eps)

        min_diff = 0  # 难度最小值
        max_diff = 1  # 难度最大值
        
        # 计算当前中心点
        center = min_diff + (max_diff - min_diff) * kwargs['t']
        
        # 计算当前标准差
        sigma = 0.5 * (1 - kwargs['t'])
        
        # 如果在最后10%的训练轮次
        if kwargs['t'] > 0.9:
            center = max_diff
            sigma = 0.1
            
        # 计算权重
        weights = torch.exp(-(difficulty - center)**2 / (2 * sigma**2))

        return weights
 
    def tokenDrugFusion(self, input_embeddings, token_mask, drugs_repr):
        """融合token表示和药物知识
        
        Args:
            input_embeddings: [B,T,N,D] batch,time,token_num,dim 
            token_mask: [B,T,N] 
            drugs_repr: [M,D] drug_num,dim
        """
        B,T,N,D = input_embeddings.shape
        M = drugs_repr.shape[0]
        
        # Mask padding tokens
        masked_embeddings = input_embeddings * token_mask.unsqueeze(-1)
        
        # Reshape token embeddings 
        token_flat = masked_embeddings.view(B*T*N, D)  # [B*T*N, D]
        
        # Calculate attention scores
        attention_scores = torch.matmul(token_flat, drugs_repr.T)  # [B*T*N, M]
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Get drug-aware token representations
        drug_context = torch.matmul(attention_weights, drugs_repr)  # [B*T*N, D]
        
        # Combine with original embeddings
        fused_embeddings = token_flat + drug_context
        
        # Reshape back
        output = fused_embeddings.view(B,T,N,D)
        
        return output * token_mask.unsqueeze(-1)   

    def process_single_view(self, input_data: List[List[List[str]]]) -> tuple:
        """Process a single view from raw input to representation
        
        Args:
            input_data: List of lists containing string tokens
            
        Returns:
            tuple: (view_repr, visit_mask)
        """
        # Encode input data
        input_ids = self.feat_tokenizers.batch_encode_3d(input_data)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        
        # Get embeddings
        input_embeddings = self.embeddings(input_ids)
        
        # Process token masks
        token_mask = (input_ids != self.feat_tokenizers.get_padding_index())
        input_embeddings = self.tokenDrugFusion(input_embeddings, token_mask, self.drugs_resp)
        
        # Sum over token dimension
        input_repr = torch.sum(input_embeddings, dim=2)
        visit_mask = torch.any(input_repr != 0, dim=2)
        
        # Get sequence encoding
        view_repr = self.seq_encoder(input_repr, visit_mask)
        
        return view_repr, visit_mask

    def contrastive_loss(self,diag_rep, proc_rep, visit_mask, temperature=0.5):
        """
        计算带有效访问掩码的对比损失
        Args:
            diag_rep: [B, V, H] 诊断表示
            proc_rep: [B, V, H] 处理表示
            visit_mask: [B, V] 访问掩码，True表示这次visit有效
            temperature: 温度参数
        Returns:
            loss: 考虑了有效样本的对比损失
            num_valid_samples: 参与计算的有效样本数
        """
        B, V, H = diag_rep.size()
        
        # 将掩码展平并获取有效样本的索引
        flat_mask = visit_mask.reshape(-1)  # [B*V]
        valid_indices = torch.where(flat_mask)[0]
        
        # 只选择有效的样本
        valid_diag_rep = diag_rep.reshape(-1, H)[valid_indices]  # [N, H], N是有效样本数
        valid_proc_rep = proc_rep.reshape(-1, H)[valid_indices]  # [N, H]
        
        # L2归一化
        valid_diag_rep = F.normalize(valid_diag_rep, dim=1)
        valid_proc_rep = F.normalize(valid_proc_rep, dim=1)
        
        # 计算有效样本间的相似度矩阵
        logits = torch.matmul(valid_diag_rep, valid_proc_rep.T) / temperature  # [N, N]
        
        # 正样本标签（对角线位置）
        labels = torch.arange(len(valid_indices), device=diag_rep.device)
        
        # 计算loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


    def calculate_loss(
            self,
            input: List[List[List[str]]],
            drugs: List[List[str]],
            mask: List[int],
            **kwargs
    ) -> tuple:
        """Calculate loss for single view data"""
        input = [input[i] for i in mask]
        drugs = [drugs[i] for i in mask]
        
        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)
        self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
        
        view_repr, visit_mask = self.process_single_view(input)
        
        out1, out2 = self.cross_attention(view_repr, view_repr)
        combined_repr = torch.cat([out1[:, -1, :], out2[:, -1, :]], dim=1)
        patient_rep = self.patient_net(combined_repr)
        
        # patient_rep = self.patient_net(view_repr[:, -1, :])
        
        logits = self.fc(patient_rep)
        
        if self.training:
            sample_weights = self.get_dynamic_weights(mask, **kwargs)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                curr_drugs,
                reduction='none'
            ).mean(dim=1)
            bce = (bce * sample_weights).sum()
        else:
            bce = F.binary_cross_entropy_with_logits(logits, curr_drugs)
            
        y_prob = torch.sigmoid(logits)
        
        return bce, y_prob, curr_drugs

    def calculate_loss_dual_view(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            mask: List[int],
            **kwargs
    ) -> tuple:
        """Calculate loss for dual view data with cross attention and contrastive learning"""
        conditions = [conditions[i] for i in mask]
        procedures = [procedures[i] for i in mask]
        drugs = [drugs[i] for i in mask]
        
        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer)
        self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
        
        # Process conditions
        cond_repr, cond_mask = self.process_single_view(conditions)
        
        # Process procedures
        proc_repr, proc_mask = self.process_single_view(procedures)
        
        # Calculate contrastive loss
        visit_mask = torch.any(cond_repr != 0, dim=2)
        cl_loss = self.contrastive_loss(cond_repr, proc_repr, visit_mask)
        
        # Apply cross attention
        cond_repr_cross, proc_repr_cross = self.cross_attention(cond_repr, proc_repr)
        combined_repr = torch.cat([cond_repr_cross[:, -1, :], proc_repr_cross[:, -1, :]], dim=1)

        patient_rep = self.patient_net(combined_repr) 
        
        logits = self.fc(patient_rep)
        
        if self.training:
            sample_weights = self.get_dynamic_weights(mask, **kwargs)
            bce = F.binary_cross_entropy_with_logits(
                logits,
                curr_drugs,
                reduction='none'
            ).mean(dim=1)
            bce = (bce * sample_weights).sum()
        else:
            bce = F.binary_cross_entropy_with_logits(logits, curr_drugs)
        
        total_loss = bce + self.gamma * cl_loss
        
        return total_loss

    def forward(
            self,
            inputs: List[List[List[str]]],
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        mask = list(range(len(inputs)))  # 所有视图的样本都取
        loss, y_prob, curr_drugs = self.calculate_loss(inputs, drugs, mask, **kwargs)
        
        # Process dual-view samples
        dual_view_mask = [i for i, v in enumerate(kwargs['view_type']) if v == 'all']
        if len(dual_view_mask) > 0:
            dual_loss= self.calculate_loss_dual_view(conditions, procedures, drugs, dual_view_mask, **kwargs)
            loss += dual_loss

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }