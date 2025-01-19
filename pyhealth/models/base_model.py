from abc import ABC, abstractmethod
import os
from typing import List, Dict, Union, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyhealth.datasets import SampleBaseDataset
from pyhealth.medcode.codes.atc import ATC
from pyhealth.models.utils import batch_to_multihot
from pyhealth.medcode.utils import download_and_read_json
from sklearn.decomposition import PCA
from pyhealth.tokenizer import Tokenizer

# TODO: add support for regression
VALID_MODE = ["binary", "multiclass", "multilabel", "regression"]


class BaseModel(ABC, nn.Module):
    """Abstract class for PyTorch models.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys: list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel". Default is None.
            Note that when mode is None, some class methods may not work (e.g.,
            `get_loss_function` and `prepare_y_prob`).
    """

    def __init__(
        self,
        dataset: SampleBaseDataset,
        feature_keys: List[str],
        label_key: str,
        mode: Optional[str] = None,
        pretrained_emb: str = None,
        process_missing_view_method: str = "zero"
    ):
        super(BaseModel, self).__init__()
        if mode is not None:
            assert mode in VALID_MODE, f"mode must be one of {VALID_MODE}"
        if process_missing_view_method not in ["copy", "avg", "diffusion","zero","random"]:
            raise ValueError(f"process_missing_view_method must be one of ['copy', 'avg', 'diffusion','zero','random']")
        
        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        self.process_missing_view_method = process_missing_view_method
        self.view_means = {}  # 存储每个视图的平均值
        self.view_counts = {} # 存储每个视图的样本数
        # pretrained embedding type, should be in ["KG", "LM", None]
        if pretrained_emb is not None:
            assert pretrained_emb[:3] in ["KG/",
                                          "LM/"], f"pretrained_emb must start with one of ['KG/', 'LM/']"
        # self.rand_init_embedding = nn.ModuleDict()
        # self.pretrained_embedding = nn.ModuleDict()
        self.pretrained_emb = pretrained_emb
        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))
        
        # save ddi adj
        self.label_tokenizer = self.get_label_tokenizer()
        ddi_adj = self.generate_ddi_adj()
        from pyhealth import BASE_CACHE_PATH as CACHE_PATH
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj)
        return

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
    
    @property
    def device(self):
        """Gets the device of the model."""
        return self._dummy_param.device

    def get_feature_tokenizers(self, special_tokens=None) -> Dict[str, Tokenizer]:
        """Gets the default feature tokenizers using `self.feature_keys`.

        These function is used for specific healthcare models, such as gamenet, safedrug, etc.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is ["<pad>", "<unk>"].

        Returns:
            feature_tokenizers: a dictionary of feature tokenizers with keys
                corresponding to self.feature_keys.
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        feature_tokenizers = {}
        for feature_key in self.feature_keys:
            feature_tokenizers[feature_key] = Tokenizer(
                tokens=self.dataset.get_all_tokens(key=feature_key),
                special_tokens=special_tokens,
            )
        return feature_tokenizers

    @staticmethod
    def get_embedding_layers(
        feature_tokenizers: Dict[str, Tokenizer],
        embedding_dim: int,
    ) -> nn.ModuleDict:
        """Gets the default embedding layers using the feature tokenizers.

        These function is used for specific healthcare models, such as gamenet, safedrug, etc.

        Args:
            feature_tokenizers: a dictionary of feature tokenizers with keys
                corresponding to `self.feature_keys`.
            embedding_dim: the dimension of the embedding.

        Returns:
            embedding_layers: a module dictionary of embedding layers with keys
                corresponding to `self.feature_keys`.
        """
        embedding_layers = nn.ModuleDict()
        for key, tokenizer in feature_tokenizers.items():
            embedding_layers[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )
        return embedding_layers

    @staticmethod
    def padding2d(batch):
        """
        Similar to pyhealth.tokenizer.Tokenizer.padding2d, but no mapping
        Args:
            batch: a list of list of list obj
                - 1-level: number of samples/patients
                - 2-level: number of visit, length maybe not equal
                - 3-level: number of features per visit, length must be equal
        Returns:
            padded_batch: a padded list of list of list obj
            e.g.,
                batch = [[[1.3, 2.5], [3.2, 4.4]], [[5.1, 6.0], [7.7, 8.3]]] -> [[[1.3, 2.5], [3.2, 4.4]], [[5.1, 6.0], [7.7, 8.3]]]
                batch = [[[1.3, 2.5], [3.2, 4.4]], [[5.1, 6.0]]] -> [[[1.3, 2.5], [3.2, 4.4]], [[5.1, 6.0], [0.0, 0.0]]]
        """
        batch_max_length = max([len(x) for x in batch])

        # get mask
        mask = torch.zeros(len(batch), batch_max_length, dtype=torch.bool)
        for i, x in enumerate(batch):
            mask[i, : len(x)] = 1

        # level-2 padding
        batch = [x + [[0.0] * len(x[0])] * (batch_max_length - len(x)) for x in batch]

        return batch, mask

    @staticmethod
    def padding3d(batch):
        """
        Similar to pyhealth.tokenizer.Tokenizer.padding2d, but no mapping
        Args:
            batch: a list of list of list obj
                - 1-level: number of samples/patients
                - 2-level: number of visit, length maybe not equal
                - 3-level: number of features per visit, length must be equal
        Returns:
            padded_batch: a padded list of list of list obj. No examples, just one more dimension higher than self.padding2d
        """
        batch_max_length_level2 = max([len(x) for x in batch])
        batch_max_length_level3 = max(
            [max([len(x) for x in visits]) for visits in batch]
        )

        # the most inner vector length
        vec_len = len(batch[0][0][0])

        # get mask
        mask = torch.zeros(
            len(batch),
            batch_max_length_level2,
            batch_max_length_level3,
            dtype=torch.bool,
        )
        for i, visits in enumerate(batch):
            for j, x in enumerate(visits):
                mask[i, j, : len(x)] = 1

        # level-2 padding
        batch = [
            x + [[[0.0] * vec_len]] * (batch_max_length_level2 - len(x)) for x in batch
        ]

        # level-3 padding
        batch = [
            [x + [[0.0] * vec_len] * (batch_max_length_level3 - len(x)) for x in visits]
            for visits in batch
        ]

        return batch, mask

    def add_feature_transform_layer(self, feature_key: str, info, special_tokens=None):
        if info["type"] == str:
            # feature tokenizer
            if special_tokens is None:
                special_tokens = ["<pad>", "<unk>"]
            tokenizer = Tokenizer(
                tokens=self.dataset.get_all_tokens(key=feature_key),
                special_tokens=special_tokens,
            )
            self.feat_tokenizers[feature_key] = tokenizer
            # feature embedding
            if self.pretrained_emb != None:
                print(f"Loading pretrained embedding for {feature_key}...")
                # load pretrained embedding
                feature_embedding_dict, special_tokens_embedding_dict \
                    = self.get_pretrained_embedding(feature_key, special_tokens, self.pretrained_emb)
                emb = []
                for i in range(tokenizer.get_vocabulary_size()):
                    idx2token = tokenizer.vocabulary.idx2token
                    if idx2token[i] in special_tokens:
                        emb.append(special_tokens_embedding_dict[idx2token[i]])
                    else:
                        emb.append(feature_embedding_dict[idx2token[i]])
                emb = torch.FloatTensor(emb)
                pretrained_emb_dim = emb.shape[1]

                self.embeddings[feature_key] = nn.Embedding.from_pretrained(
                    emb,
                    padding_idx=tokenizer.get_padding_index(),
                    freeze=False,
                )

                self.linear_layers[feature_key] = nn.Linear(pretrained_emb_dim , self.embedding_dim)

                # Compute PCA on embeddings
                # pca = PCA(n_components=self.embedding_dim)
                # pca.fit(emb.numpy()) # assumes emb is a torch tensor

                # # Use the PCA to transform embeddings
                # transformed_emb = pca.transform(emb.numpy())

                # # Then load these transformed embeddings into a new nn.Embedding layer
                # self.embeddings[feature_key] = nn.Embedding.from_pretrained(
                #     torch.tensor(transformed_emb),
                #     padding_idx=tokenizer.get_padding_index(),
                #     freeze=False,
                # )

            else:
                self.embeddings[feature_key] = nn.Embedding(
                    tokenizer.get_vocabulary_size(),
                    self.embedding_dim,
                    padding_idx=tokenizer.get_padding_index(),
                )
        elif info["type"] in [float, int]:
            self.linear_layers[feature_key] = nn.Linear(info["len"], self.embedding_dim)
        else:
            raise ValueError("Unsupported feature type: {}".format(info["type"]))

    def get_pretrained_embedding(self, feature_key: str, special_tokens=None, pretrained_type="LM/clinicalbert"):
        feature_embedding_file = f"embeddings/{pretrained_type}/{feature_key}/{self.dataset.code_vocs[feature_key].lower()}.json"
        feature_embedding = download_and_read_json(feature_embedding_file)

        if special_tokens is not None:
            special_tokens_embedding_file = f"embeddings/{pretrained_type}/special_tokens/special_tokens.json"
            special_tokens_embedding = download_and_read_json(special_tokens_embedding_file)
        else:
            special_tokens_embedding = None
        
        return feature_embedding, special_tokens_embedding

    def get_label_tokenizer(self, special_tokens=None) -> Tokenizer:
        """Gets the default label tokenizers using `self.label_key`.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is empty list.

        Returns:
            label_tokenizer: the label tokenizer.
        """
        if special_tokens is None:
            special_tokens = []
        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key),
            special_tokens=special_tokens,
        )
        return label_tokenizer

    def get_output_size(self, label_tokenizer: Tokenizer) -> int:
        """Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Args:
            label_tokenizer: the label tokenizer.

        Returns:
            output_size: the output size of the model.
        """
        output_size = label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def get_loss_function(self) -> Callable:
        """Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`

        Returns:
            The default loss function.
        """
        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "regression":
            return F.mse_loss
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_labels(
        self,
        labels: Union[List[str], List[List[str]]],
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
        """Prepares the labels for model training and evaluation.

        This function converts the labels to different formats depending on the
        mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1)
            - multiclass: a tensor of shape (batch_size,)
            - multilabel: a tensor of shape (batch_size, num_labels)

        Args:
            labels: the raw labels from the samples. It should be
                - a list of str for binary and multiclass classificationa
                - a list of list of str for multilabel classification
            label_tokenizer: the label tokenizer.

        Returns:
            labels: the processed labels.
        """
        if self.mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif self.mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.to(self.device)
        return labels

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        depending on the mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`

        Args:
            logits: the predicted logit tensor.

        Returns:
            y_prob: the predicted probability tensor.
        """
        if self.mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif self.mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif self.mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        else:
            raise NotImplementedError
        return y_prob
    
    def forward_total_view(
            self,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        conditions_ids = torch.tensor(conditions, dtype=torch.long, device=self.device)
        conditions = self.embeddings['conditions'](conditions_ids)
        if getattr(self, 'is_drug_fuse', False):
            token_mask = (conditions_ids != 0)
            self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
            conditions = self.tokenDrugFusion(conditions, token_mask, self.drugs_resp)
        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures_ids = torch.tensor(procedures, dtype=torch.long, device=self.device)
        procedures = self.embeddings['procedures'](procedures_ids)
        if getattr(self, 'is_drug_fuse', False):
            token_mask = (procedures_ids != 0)
            self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
            procedures = self.tokenDrugFusion(procedures, token_mask, self.drugs_resp)
        procedures = torch.sum(procedures, dim=2)

        return conditions,procedures,mask
    
    def caculate_avg_view_representation(
        self, 
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs
    ):  
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        conditions_ids = torch.tensor(conditions, dtype=torch.long, device=self.device)
        conditions = self.embeddings['conditions'](conditions_ids)
        if getattr(self, 'is_drug_fuse', False):
            token_mask = (conditions_ids != 0)
            self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
            conditions = self.tokenDrugFusion(conditions, token_mask, self.drugs_resp)
        conditions = torch.sum(conditions, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures_ids = torch.tensor(procedures, dtype=torch.long, device=self.device)
        procedures = self.embeddings['procedures'](procedures_ids)
        if getattr(self, 'is_drug_fuse', False):
            token_mask = (procedures_ids != 0)
            self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
            procedures = self.tokenDrugFusion(procedures, token_mask, self.drugs_resp)
        procedures = torch.sum(procedures, dim=2)

        view_rep = {}
        view_rep["conditions"] = conditions
        view_rep["procedures"] = procedures

        for view_name in self.feature_keys:
            current_sum = view_rep[view_name].sum(dim=0)  # 当前batch的和
            current_count = view_rep[view_name].size(0)   # 当前batch的样本数
            
            if view_name not in self.view_means:
                self.view_means[view_name] = current_sum
                self.view_counts[view_name] = current_count
            else:
                # 在线更新平均值
                total_count = self.view_counts[view_name] + current_count
                self.view_means[view_name] = (self.view_means[view_name] * self.view_counts[view_name] + current_sum) / total_count
                self.view_counts[view_name] = total_count


    def get_avg_view_representation(self, view_name: str):
        if view_name not in self.feature_keys:
            raise ValueError(f"{view_name} is not in feature_keys")
        if view_name not in self.view_means:
            raise ValueError(f"{view_name} is not in view_means")
        return self.view_means[view_name]
    
    def forward_single_view(
        self,
        input_view_name: str,
        missing_view_name: str,
        input_view: List[List[List[str]]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_view = self.feat_tokenizers[input_view_name].batch_encode_3d(input_view)
        input_ids = torch.tensor(input_view, dtype=torch.long, device=self.device)
        input_view = self.embeddings[input_view_name](input_ids)
        if getattr(self, 'is_drug_fuse', False):
            token_mask = (input_ids != 0)
            self.drugs_resp = (self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.x)).to(self.device)
            input_view = self.tokenDrugFusion(input_view, token_mask, self.drugs_resp)

        input_view = torch.sum(input_view, dim=2)
        mask = torch.any(input_view != 0, dim=2)

        # 方法1，使用另一个的视图来填充
        generated_view = None
        if self.process_missing_view_method == "copy":
            generated_view = input_view
        # 方法2，使用平均表征模型来生成
        elif self.process_missing_view_method == "avg":
            generated_view = self.get_avg_view_representation(missing_view_name)[:input_view.shape[1],:].unsqueeze(0).repeat(input_view.shape[0], 1, 1)
        # 方法3，使用0向量来填充
        elif self.process_missing_view_method == "zero":
            generated_view = torch.zeros_like(input_view)
        # 方法4，使用随机向量来填充
        elif self.process_missing_view_method == "random":
            generated_view = torch.randn_like(input_view)
        
        if input_view_name == "conditions":
            return input_view,generated_view,mask
        elif input_view_name == "procedures":
            return generated_view,input_view,mask
    
    def get_patient_view_representation(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if "conditions" in  kwargs['input_keys'] and "procedures" in  kwargs['input_keys']:
            conditions,procedures,mask = self.forward_total_view(conditions, procedures, **kwargs)
        elif "conditions" in  kwargs['input_keys']:
            conditions,procedures,mask = self.forward_single_view(input_view_name='conditions', missing_view_name='procedures', input_view=conditions,  **kwargs)
        elif "procedures" in  kwargs['input_keys']:
            conditions,procedures,mask = self.forward_single_view(input_view_name='procedures', missing_view_name='conditions', input_view=procedures,  **kwargs)

        return conditions,procedures,mask