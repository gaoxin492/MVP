import json
import os
import glob
import random
import numpy as np
import torch
import copy
import math
from cyclic_permutation import *
from complete_shuffle_package import *
from tqdm import tqdm



def total_permutation(masks: list, num: int):
    """
    Generate all possible permutations of a binary vector.
    """
    idx = [i for i, mask in enumerate(masks) if mask == 1]
    num_modalities = len(masks)
    random.seed(42)
    all_permutations = double_cyclic_permutation(idx, num)
    # all_permutations = non_cyclic_permutation(idx, num)

    results = []
    for perm in all_permutations:
        result = list(range(num_modalities))
        for i, p in enumerate(perm):
            result[idx[i]] = p
        results.append(result)
        
    return results

def generate_fingerprint(data_path, num_modalities, missing_rate, num_samples, num, seed):

    fingerprint_json = {i: {} for i in range(num_samples)}
    random.seed(seed)
    torch.manual_seed(seed)

    index = [i for i in range(num_samples)]
    random.shuffle(index)
    index_miss = index[:int(num_samples*missing_rate)]
    index_preserve = index[int(num_samples*missing_rate):]
    print("Length of missing data: ", len(index_miss))
    print("Length of preserved data: ", len(index_preserve))

    for mv_samples in tqdm(index_miss):
        # 生成列表长度为num_modalities的随机二进制列表
        masks = torch.randint(0, 2, (num_modalities,))
        # 生成的masks列表中全为1或者全为0，重新生成
        while torch.sum(masks) == num_modalities or torch.sum(masks) == 0:
            masks = torch.randint(0, 2, (num_modalities,))
        fingerprint_json[mv_samples]['mask'] = masks.tolist()
        all_permutations = total_permutation(masks.tolist(), num)
        fingerprint_json[mv_samples]['pt'] = all_permutations
    
    masks = torch.ones(num_modalities)
    all_permutations = total_permutation(masks.tolist(), num)
    for mv_samples in tqdm(index_preserve):

        fingerprint_json[mv_samples]['mask'] = masks.tolist()
        fingerprint_json[mv_samples]['pt'] = all_permutations

    # return fingerprint_json
    json_name = 'fingerprint' + str(num_views) + '_' + str(int(10*missing_rate)) + '.json'
    save_path = os.path.join(data_path, json_name)
    with open(save_path, 'w') as f:
        json.dump(fingerprint_json, f, indent=4)


num_views = 6
num_samples = 2000
dataset = 'Handwritten'
data_path = '/path/to/data/'+dataset
num_pt = max(num_views, math.factorial(num_views-1))

for missing_rate in [0.5]:
    generate_fingerprint(data_path, num_views, missing_rate, num_samples, num=num_pt, seed=42)

# CUB: 2, 600
# Scene15: 3, 4485
# Handwritten: 6, 2000
# Caltech101_20: 6, 2386
# Caltech101: 6, 9144
# Reuters10: 5, 18758
# LandUse21: 3, 2100
# Leaves: 3, 1600
# Sensit: 2, 98528
# Animal: 2, 10158