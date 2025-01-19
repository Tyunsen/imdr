import os
from pathlib import Path
import time
from typing import List, Set
import random
import pickle

import numpy as np
import torch
import yaml
from pyhealth import BASE_CACHE_PATH
from pyhealth.datasets.mimic3 import MIMIC3Dataset
from pyhealth.medcode.inner_map import InnerMap
# from pyhealth.tasks.drug_recommendation import drug_recommendation_mimic3_fn
from typing import List, Tuple, Set
import heapq
import torch.nn.functional as F
ROOT_DIR = Path(__file__).parent 
with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as file:
    config = yaml.safe_load(file)


def history2instace(patients: List[List[List[int]]]):
    batch = []
    for patient in patients:
        for visit in patient:
            if len(visit)>0:
                batch.append(visit)
    return batch

def generate_random_seed():
    return random.randint(0, 2 ** 32 - 1) % 10000000000

def tranfer3Dto2DList(input:List[List[List[str]]]):
    res = []
    for patient in input:
        res.append(patient[-1])
    return res
        


