from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import sys
import os
import yaml
import pickle
from pyhealth.datasets.splitter import split_views
from pyhealth.utils import set_seed
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks import drug_recommendation_mimic3_fn,drug_recommendation_mimic4_fn
from trainer import Trainer
from model import IMDR 
from pathlib import Path
from torch.utils.data import ConcatDataset 
from pyhealth.metrics.drug_recommendation import ddi_rate_score
from gen_token_complexity import calculate_sample_complexity
# 读取yaml文件
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

def load_datasets(missing_rate):
    '''
        获取或创建数据集分割
        missing_rate: 缺失率
    '''
    # 创建缓存目录
    cache_dir = os.path.join('.', "data",config['dataset'],f"missing_rate_{missing_rate}")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"缓存目录为：{cache_dir}")   

    # 构建缓存文件路径
    cache_files = {
        'sample_dataset': f'{cache_dir}/sample_dataset.pkl',
        'total_train': f'{cache_dir}/total_view_train.pkl',
        'view1_train': f'{cache_dir}/only_view1_train.pkl',
        'view2_train': f'{cache_dir}/only_view2_train.pkl',
        'total_val': f'{cache_dir}/total_view_val.pkl',
        'view1_val': f'{cache_dir}/only_view1_val.pkl',
        'view2_val': f'{cache_dir}/only_view2_val.pkl',
        'total_test': f'{cache_dir}/total_view_test.pkl',
        'view1_test': f'{cache_dir}/only_view1_test.pkl',
        'view2_test': f'{cache_dir}/only_view2_test.pkl'
    }
    
    # 检查是否所有缓存文件都存在
    all_cached = all(os.path.exists(f) for f in cache_files.values())
    
    if all_cached:
        print("从缓存加载数据集分割...")
        datasets = {}
        for key, path in cache_files.items():
            with open(path, 'rb') as f:
                datasets[key] = pickle.load(f)
        return (datasets['sample_dataset'],datasets['total_train'], datasets['view1_train'], datasets['view2_train'],
                datasets['total_val'], datasets['view1_val'], datasets['view2_val'],
                datasets['total_test'], datasets['view1_test'], datasets['view2_test'])
    
    print("创建新的数据集分割...")
    # 创建数据集分割
    if config['dataset'] == "MIMIC3":
        base_dataset = MIMIC3Dataset(
            root=config['mimiciii_path'],
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=config['dev'],
            refresh_cache=config['refresh_cache']
        )
        sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
    elif config['dataset'] == "MIMIC4":
        base_dataset = MIMIC4Dataset(
            root=config['mimiciv_path'],
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=config['dev'],
            refresh_cache=config['refresh_cache']
        )
        sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn)

    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.7, 0.15, 0.15])
    # 再次切分 train_dataset
    total_view_train, only_view1_train, only_view2_train = split_views(sample_dataset,train_dataset.indices, missing_rate)
    total_view_val, only_view1_val, only_view2_val = split_views(sample_dataset,val_dataset.indices ,missing_rate)
    total_view_test, only_view1_test, only_view2_test = split_views(sample_dataset, test_dataset.indices,missing_rate)

    # 保存到缓存
    
    datasets = {
        'sample_dataset': sample_dataset,
        'total_train': total_view_train,
        'view1_train': only_view1_train,
        'view2_train': only_view2_train,
        'total_val': total_view_val,
        'view1_val': only_view1_val,
        'view2_val': only_view2_val,
        'total_test': total_view_test,
        'view1_test': only_view1_test,
        'view2_test': only_view2_test
    }
    
    for key, dataset in datasets.items():
        with open(cache_files[key], 'wb') as f:
            pickle.dump(dataset, f)
    
    return (sample_dataset,total_view_train, only_view1_train, only_view2_train,
            total_view_val, only_view1_val, only_view2_val,
            total_view_test, only_view1_test, only_view2_test)

def set_input_views(model,dataset,view_type):
    for sample in tqdm(dataset):
        view_score = 0.5

        curr_drugs = model.prepare_labels([sample['drugs']], model.label_tokenizer)
        y_pred = curr_drugs.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        current_ddi_rate = ddi_rate_score(y_pred, model.ddi_adj.cpu().numpy())

        if view_type == 'all':
            sample['view_score'] = 1+config['IMDR']['epsilon']
            visits = []
            for diag,proc in zip(sample['conditions'],sample['procedures']):
                visits.append(diag+proc)
            sample['inputs'] = visits
        elif view_type == 'diag':
            sample['view_score'] = config['IMDR']['epsilon']
            sample['inputs'] = sample['conditions']
        elif view_type == 'proc':
            sample['view_score'] = config['IMDR']['epsilon']
            sample['inputs'] = sample['procedures']

        sample['view_type'] = view_type
        sample['ddi_rate'] = current_ddi_rate
        sample['complexity'] = calculate_sample_complexity(sample['drugs'],sample['conditions'][-1],sample['procedures'][-1],datasetname=config['dataset'])
        sample['tokens_num'] = len(sample['inputs'][-1])
    return dataset

def run(model_name,missing_rate):

    # STEP 1: load data
    (sample_dataset,total_view_train_dataset, only_view1_train_dataset, only_view2_train_dataset,
    total_view_val_dataset, only_view1_val_dataset, only_view2_val_dataset,
    total_view_test_dataset, only_view1_test_dataset, only_view2_test_dataset) = load_datasets(
        missing_rate
    )

    model = IMDR(
            sample_dataset,
            embedding_dim=config['embedding_dim'],
            alpha=config['IMDR']['alpha'],
            beta=config['IMDR']['beta'],
        )

    set_input_views(model,total_view_train_dataset,view_type='all') 
    set_input_views(model,total_view_val_dataset,view_type='all')
    set_input_views(model,total_view_test_dataset,view_type='all') 
    set_input_views(model,only_view1_train_dataset,view_type='diag') 
    set_input_views(model,only_view1_val_dataset,view_type='diag') 
    set_input_views(model,only_view1_test_dataset,view_type='diag') 
    set_input_views(model,only_view2_train_dataset,view_type='proc') 
    set_input_views(model,only_view2_val_dataset,view_type='proc')
    set_input_views(model,only_view2_test_dataset,view_type='proc')

    # 合并训练集
    combined_train_dataset = ConcatDataset([total_view_train_dataset, only_view1_train_dataset, only_view2_train_dataset])  # 合并数据集
    combined_val_dataset = ConcatDataset([total_view_val_dataset, only_view1_val_dataset, only_view2_val_dataset])
    combined_test_dataset = ConcatDataset([total_view_test_dataset, only_view1_test_dataset, only_view2_test_dataset])
    
    set_seed(config['seed'])

    # 获取dataloader
    train_dataloader = get_dataloader(combined_train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = get_dataloader(combined_val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = get_dataloader(combined_test_dataset, batch_size=config['batch_size'], shuffle=False)    
    
    # STEP 2: define trainer
    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "pr_auc_samples", "f1_samples","roc_auc_samples","ddi_score", "avg_med"],
        device=config['device'],
        model_name=f"{model_name}",
        seed=config['seed'],
    )

    # STEP 3: train
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=config['epochs'],
        monitor="jaccard_samples",
        lr=config['lr']
    )


if __name__ == "__main__":

    for missing_rate in config['missing_rates']:
        run(model_name=config['model'],missing_rate=missing_rate)
    

