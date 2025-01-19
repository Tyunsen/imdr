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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Attention(nn.Module):
    def __init__(self, device, embed_dim=64, output_dim=1):
        super(Attention, self).__init__()
        self.embedding_dim, self.output_dim = embed_dim, output_dim
        self.aggregation = nn.Linear(self.embedding_dim, self.output_dim)
        self.device = device

    def _aggregate(self, x):
        weight = self.aggregation(x)  # [b, num_learn, 1]
        return torch.tanh(weight)

    def forward(self, x, mask=None):
        device = self.device

        if mask is None:
            weight = torch.softmax(self._aggregate(x), dim=-2)
        else:
            mask = torch.where(mask == 0, torch.tensor(-1e7).to(device), torch.tensor(0.0).to(device))
            weight = torch.softmax(self._aggregate(x).squeeze(-1) + mask, dim=-1).float().unsqueeze(-1)
            weight = torch.where(torch.isnan(weight), torch.tensor(0.0).to(device), weight)
        agg_embeds = torch.matmul(x.transpose(-1, -2).float(), weight).squeeze(-1)
        return agg_embeds
       
class FourSdrug(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        super(FourSdrug, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel"
        )
        self.feature_key = 'conditions' # conditions procedures
        device = kwargs['device']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.delta = kwargs['delta']

        n_sym = self.get_feature_tokenizers()[self.feature_key].get_vocabulary_size()
        n_drug = self.get_label_tokenizer().get_vocabulary_size()
        

        self.n_sym, self.n_drug = n_sym, n_drug
        self.embed_dim, self.dropout = embedding_dim, 0.4
        self.sym_embeddings = nn.Embedding(self.n_sym, self.embed_dim,padding_idx=self.get_feature_tokenizers()[self.feature_key].get_padding_index())
        self.drug_embeddings = nn.Embedding(self.n_drug, self.embed_dim)
        self.sym_agg = Attention(device,self.embed_dim)
        self.tensor_ddi_adj = self.generate_ddi_adj().to(device)
        self.init_parameters()

        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.generate_ddi_adj().cpu())


    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.get_label_tokenizer().get_vocabulary_size()
        vocab_to_index = self.get_label_tokenizer().vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def find_similae_set_by_ja(self, syms:List[List[str]]) -> List[int]:
        '''
        :param syms: [batch_size, sym_set_size]
        :return: similar_idx: [batch_size]
        '''
        def jaccard_scores(set1, set2):
            """计算两个集合的 Jaccard 相似度"""
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0

        similar_idx = []
        for i, sym_set in enumerate(syms):
            max_similarity = 0
            best_index = i
            current_set = set(sym_set)
            
            for j, other_set in enumerate(syms):
                if i == j:
                    continue
                similarity = jaccard_scores(current_set, set(other_set))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_index = j
            
            similar_idx.append(best_index)

        return torch.tensor(similar_idx,dtype=torch.int64,device=self.device)
    
    def forward(self, drugs:List[List[str]],**kwargs):
        '''
        :param syms: [batch_size, sym_set_size] 
        :param drugs: [batch_size, num_drugs]
        :param similar_idx: [batch_size]
        :return:
        '''
        syms = kwargs[self.feature_key]
        syms = [patient[-1] for patient in syms]
        device = self.device
        similar_idx = self.find_similae_set_by_ja(syms)
        drugs = self.prepare_labels(drugs, self.get_label_tokenizer()).to(dtype=torch.float64)
        syms = self.get_feature_tokenizers()[self.feature_key].batch_encode_2d(syms)
        syms = torch.tensor(syms,dtype=torch.int64,device=device)

        all_drugs = torch.tensor(range(self.n_drug)).to(device)
        sym_embeds, all_drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(all_drugs)
        s_set_embeds = self.sym_agg(sym_embeds)
        # s_set_embeds = torch.mean(sym_embeds, dim=1)
        all_drug_embeds = all_drug_embeds.unsqueeze(0).repeat(s_set_embeds.shape[0], 1, 1)

        scores = torch.bmm(s_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(-2)  # [batch_size, n_drug]
        scores_aug, batch_neg = 0.0, 0.0

        neg_pred_prob = torch.sigmoid(scores)

        neg_pred_prob = torch.mm(neg_pred_prob.transpose(-1, -2), neg_pred_prob)  # (voc_size, voc_size)
        # t = 0.00001 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        batch_neg = (neg_pred_prob.mul(self.tensor_ddi_adj).sum()) / self.tensor_ddi_adj.shape[0] ** 2

        if syms.shape[0] > 2 and syms.shape[1] > 2:
            scores_aug = self.intraset_augmentation(syms, drugs, all_drug_embeds, similar_idx)
            intersect_ddi = self.intersect_ddi(syms, s_set_embeds, drugs, all_drug_embeds, similar_idx)
            batch_neg += intersect_ddi

        sig_scores = torch.sigmoid(scores)
        scores_sigmoid = torch.where(sig_scores == 0, torch.tensor(1.0).to(device), sig_scores)

        bce_loss = F.binary_cross_entropy_with_logits(scores, drugs)
        entropy = -torch.mean(sig_scores * (torch.log(scores_sigmoid) - 1))
        loss = bce_loss + self.beta * entropy + self.gamma * scores_aug + self.delta * batch_neg
        y_prob = torch.sigmoid(scores)
        
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": drugs
        }

    def intraset_augmentation(self, syms, drugs, all_drug_embeds, similar_idx):
        device = self.device

        selected_drugs = drugs[similar_idx]
        r = torch.tensor(range(drugs.shape[0])).to(device).unsqueeze(1)
        sym_multihot, selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym)).to(device), \
                                              torch.zeros((drugs.shape[0], self.n_sym)).to(device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.tensor(range(self.n_sym)).to(device)).unsqueeze(0).expand_as(common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        common_drug, diff_drug = drugs * selected_drugs, drugs - selected_drugs
        # diff_drug[diff_drug == -1] = 1

        # common_drug_sum, diff_drug = torch.sum(common_drug, -1, True), torch.sum(diff_drug, -1, True)
        # common_drug_sum[common_drug_sum == 0], diff_drug[diff_drug == 0] = 1, 1

        scores = torch.bmm(common_set_embeds.unsqueeze(1), all_drug_embeds.transpose(-1, -2)).squeeze(1)
        scores = F.binary_cross_entropy_with_logits(scores, common_drug)

        return scores

    def intersect_ddi(self, syms, s_set_embed, drugs, all_drug_embeds, similar_idx):
        device = self.device

        selected_drugs = drugs[similar_idx]
        r = torch.tensor(range(drugs.shape[0])).to(device).unsqueeze(1)
        sym_multihot, selected_sym_multihot = torch.zeros((drugs.shape[0], self.n_sym)).to(device), \
                                              torch.zeros((drugs.shape[0], self.n_sym)).to(device)
        sym_multihot[r, syms], selected_sym_multihot[r, syms[similar_idx]] = 1, 1

        common_sym = sym_multihot * selected_sym_multihot
        common_sym_sq = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds = self.sym_embeddings(torch.tensor(range(self.n_sym)).to(device)).unsqueeze(0).expand_as(
            common_sym_sq)
        common_sym_embeds = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        diff_drug = drugs - selected_drugs
        diff_drug_2 = torch.zeros_like(diff_drug)
        diff_drug_2[diff_drug == -1], diff_drug[diff_drug == -1] = 1, 0

        diff_drug_exp, diff2_exp = diff_drug.unsqueeze(1), diff_drug_2.unsqueeze(1)
        diff_drug = torch.sum(diff_drug, -1, True)
        diff_drug_2 = torch.sum(diff_drug_2, -1, True)
        diff_drug[diff_drug == 0] = 1
        diff_drug_2[diff_drug_2 == 0] = 1
        diff_drug_embed = torch.bmm(diff_drug_exp.float(), all_drug_embeds).squeeze() / diff_drug
        diff2_embed = torch.bmm(diff2_exp.float(), all_drug_embeds).squeeze() / diff_drug_2

        diff_score = torch.sigmoid(common_set_embeds * diff_drug_embed.float())
        diff2_score = torch.sigmoid(common_set_embeds * diff2_embed.float())
        score_aug = 0.0001 * torch.sum(diff2_score * diff_score)

        return score_aug





        
    
