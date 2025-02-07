import math
import statistics
import sys
import os
from pathlib import Path
from tqdm import tqdm
import yaml
from openai import OpenAI
import time
import pickle
from pyhealth.datasets.mimic3 import MIMIC3Dataset
from pyhealth.tasks.drug_recommendation import drug_recommendation_mimic3_fn
import math
from pathlib import Path
import os
import pickle

def init_openai_client():
    return OpenAI(
        base_url="xxx",
        api_key="sk-xxx"
    )

def get_complexity_score(term, code_type, inner_map, client):
    """Get complexity score for medical codes"""
    try:
        # Get description for the code
        description = inner_map.lookup(term)
        
        if code_type == "ATC":
            prompt = f"""Rate the complexity of the following medication on a scale of 1-10, considering:
            - Administration complexity
            - Side effect profile
            - Monitoring requirements
            - Drug interactions potential
            - Treatment duration
            
            Medication: {description}
            
            Return only a number between 1-10, where:
            1: Simple, safe, minimal monitoring
            10: Complex, high-risk, intensive monitoring required"""
            
        elif code_type == "ICD9CM":
            prompt = f"""Rate the severity and complexity of the following medical condition on a scale of 1-10, considering:
            - Disease severity
            - Treatment complexity
            - Complication risks
            - Long-term impact
            - Resource requirements
            
            Condition: {description}
            
            Return only a number between 1-10, where:
            1: Minor, self-limiting condition
            10: Critical, life-threatening condition"""
            
        else:  # ICD9PROC
            prompt = f"""Rate the complexity of the following medical procedure on a scale of 1-10, considering:
            - Technical difficulty
            - Risk level
            - Resource requirements
            - Recovery complexity
            - Potential complications
            
            Procedure: {description}
            
            Return only a number between 1-10, where:
            1: Simple, routine procedure
            10: Complex, high-risk procedure"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the most capable model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 1), 10)  # 确保分数在1-10之间
        
    except Exception as e:
        print(f"获取'{term}'的评分时出错: {e}")
        return 5  # 默认中等复杂度

def load_complexity_scores(voc, code_type, retry_delay=1, datasetname="MIMIC3"):
    """加载或生成医疗代码的复杂度评分"""
    from pyhealth.medcode import InnerMap
    
    code_map = {
        "atc": ("ATC", lambda x: x),
        "icd9cm": ("ICD9CM", lambda x: x),
        "icd9proc": ("ICD9PROC", lambda x: x[:2] if len(x) > 2 else x)
    }

    if code_type not in code_map:
        raise ValueError(f"不支持的编码类型: {code_type}")

    inner_map_name, fallback_func = code_map[code_type]
    inner_map = InnerMap.load(inner_map_name)
    
    # 缓存文件路径
    file_path = os.path.join(str(Path.home()), ".cache", "pyhealth", "medcode", 
                            f"{datasetname}-{code_type}-complexity.pkl")
    
    # 加载已有的评分
    token2score = {}
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            token2score = pickle.load(f)
    
    # 初始化OpenAI客户端
    client = init_openai_client()
    
    # 记录需要处理的tokens
    new_tokens = [token for token in voc if token not in token2score]
    total_new = len(new_tokens)
    
    # 批量获取新token的评分
    for i, token in enumerate(tqdm(new_tokens, desc="Processing tokens")):
        attempt = 1
        while True:
            try:
                score = get_complexity_score(token, inner_map_name, inner_map, client)
                token2score[token] = score
                
                # 每处理100个token或出现异常时保存一次
                if (i + 1) % 100 == 0:
                    print(f"\nSaving checkpoint after processing {i + 1}/{total_new} tokens...")
                    with open(file_path, "wb") as f:
                        pickle.dump(token2score, f)
                
                break
            except Exception as e:
                print(f"\n获取'{token}'的评分时出错 (尝试 #{attempt}): {e}")
                
                # 如果出现异常，保存当前进度
                if token2score:
                    print("Saving current progress before retry...")
                    with open(file_path, "wb") as f:
                        pickle.dump(token2score, f)
                
                print(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
                attempt += 1
                
                # 如果重试超过5次，跳过当前token
                if attempt > 5:
                    print(f"Skipping token {token} after 5 failed attempts")
                    break
        
    # 最终保存
    print("\nSaving final results...")
    with open(file_path, "wb") as f:
        pickle.dump(token2score, f)
    
    print(f"{code_type.upper()}复杂度评分已保存到: {file_path}")
    return token2score


import os
from pathlib import Path
import pickle
import math
from functools import lru_cache

# 全局缓存字典
_score_cache = {}

def load_scores(datasetname):
    """加载并缓存评分数据"""
    if datasetname in _score_cache:
        return _score_cache[datasetname]
        
    cache_dir = os.path.join(str(Path.home()), ".cache", "pyhealth", "medcode")
    
    # 定义需要加载的文件
    score_files = {
        'drug': f"{datasetname}-atc-complexity.pkl",
        'condition': f"{datasetname}-icd9cm-complexity.pkl",
        'procedure': f"{datasetname}-icd9proc-complexity.pkl"
    }
    
    # 加载所有评分文件
    scores = {}
    for category, filename in score_files.items():
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                scores[category] = pickle.load(f)
        else:
            scores[category] = {}
            
    # 存入缓存
    _score_cache[datasetname] = scores
    return scores

def get_complexity_score(sequence_tuple, token2score):
    """计算复杂度分数(使用缓存)"""
    sequence = list(sequence_tuple)
    if not sequence:
        return 0.0
        
    # 计算平均严重程度分数
    severity_scores = [token2score.get(token, 5.0) for token in sequence]
    avg_severity = sum(severity_scores) / len(severity_scores)
    
    # 计算长度因子
    length_multiplier = math.log2(len(sequence) + 1)
    
    return avg_severity * length_multiplier

def calculate_sample_complexity(drugs, conditions, procedures, datasetname="MIMIC3"):
    """
    计算患者样本的复杂度分数
    
    Args:
        drugs: 药物代码列表
        conditions: 疾病代码列表
        procedures: 手术代码列表
        datasetname: 数据集名称
    
    Returns:
        float: 样本的复杂度分数
    """
    # 获取或加载评分数据
    scores = load_scores(datasetname)
    
    # 转换为元组以支持缓存
    drugs_tuple = tuple(drugs)
    conditions_tuple = tuple(conditions)
    procedures_tuple = tuple(procedures)
    
    # 计算各组件分数
    drug_score = get_complexity_score(drugs_tuple, scores['drug'])
    condition_score = get_complexity_score(conditions_tuple, scores['condition']) 
    procedure_score = get_complexity_score(procedures_tuple, scores['procedure'])
    
    # 计算最终分数
    final_score = (drug_score + condition_score + procedure_score) / 3
    
    return final_score

# 使用示例
if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    if config['dataset'] == 'MIMIC3':
        base_dataset = MIMIC3Dataset(
            root=config['mimiciii_path'],
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=config['dev'],
            refresh_cache=config['refresh_cache'],
        )
        base_dataset.stat()
        sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
    elif config['dataset'] == 'MIMIC4':
        base_dataset = MIMIC4Dataset(
            root=config['mimiciv_path'],
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            dev=config['dev'],
            refresh_cache=config['refresh_cache'],
        )
        base_dataset.stat()
        sample_dataset = base_dataset.set_task(drug_recommendation_mimic4_fn)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    
    sample_dataset.stat()
    
    # 使用方式与原代码类似
    drug_scores = load_complexity_scores(sample_dataset.get_all_tokens(key='drugs'), "atc")
    condition_scores = load_complexity_scores(sample_dataset.get_all_tokens(key='conditions'), "icd9cm")
    procedure_scores = load_complexity_scores(sample_dataset.get_all_tokens(key='procedures'), "icd9proc")
