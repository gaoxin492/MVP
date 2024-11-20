import json
import os
import glob
import random
import numpy as np
import torch
import copy
from cyclic_permutation import *
from complete_shuffle_package import *
from tqdm import tqdm
import math

def total_permutation(masks: list, num: int):
    """
    Generate all possible permutations of a binary vector.
    """
    # idx为masks列表为1的索引
    idx = np.nonzero(np.array(masks))[0].tolist()
    num_modalities = len(masks)
    random.seed(42)
    all_permutations = double_cyclic_permutation(idx, num)

    results = []
    # print(idx)
    for i in range(num):
        idx_permutations = all_permutations[i]
        # print(idx_permutations)
        idx_permutations_out = list(range(num_modalities))
        for j in range(len(idx_permutations)):
            idx_permutations_out[idx[j]] = idx_permutations[j]
        results.append(idx_permutations_out)

    # print("*"*20)

    return results

def generate_fingerprint(data_path, subset, num_modalities, missing_rate, num, seed):

    data = []
    for category in ['0', '1', '2', '3', '4']:
        data_path_cat = os.path.join(data_path, category)
        name = os.path.join('MVShapeNet', subset, category)
        for sample in os.listdir(data_path_cat):
            data_path_sample = os.path.join(data_path_cat, sample)
            data.append(os.path.join(name, sample))
    num_samples = len(data)
    print("Length of all data: ", num_samples)

    fingerprint_json = {data[i]: {} for i in range(len(data))}
    random.seed(seed)
    torch.manual_seed(seed)

    random.shuffle(data)
    index_miss = data[:int(num_samples*missing_rate)]
    index_preserve = data[int(num_samples*missing_rate):]
    print("Length of missing data: ", len(index_miss))
    print("Length of preserved data: ", len(index_preserve))

    for mv_samples in tqdm(index_miss):
        # 生成列表长度为num_modalities的随机二进制列表
        masks = torch.randint(0, 2, (num_modalities,))
        # 生成的masks列表中全为1或者全为0，重新生成
        while torch.sum(masks) == num_modalities or torch.sum(masks) == 0:
            masks = torch.randint(0, 2, (num_modalities,))
        fingerprint_json[mv_samples]['mask'] = masks.tolist()
        fingerprint_json[mv_samples]['pt'] = total_permutation(masks.tolist(), num)

    masks = torch.ones(num_modalities)
    all_permutations = total_permutation(masks.tolist(), num)
    for mv_samples in tqdm(index_preserve):
        fingerprint_json[mv_samples]['mask'] = masks.tolist()
        fingerprint_json[mv_samples]['pt'] = all_permutations

    # return fingerprint_json
    
    save_path = os.path.join(data_path, 'fingerprint' + str(num_modalities) + '_' + str(int(10*missing_rate)) + '.json')
    with open(save_path, 'w') as f:
        json.dump(fingerprint_json, f, indent=4)

num_views = 5
num_pt = max(num_views, math.factorial(num_views-1))

data_path = '/path/to/MVShapeNet'
for missing_rate in tqdm([0.5]):
    data_path_subset = os.path.join(data_path, 'train')
    generate_fingerprint(data_path_subset, 'train', 5, missing_rate, num=num_pt, seed=42)

# Scene15: 3, 4485
# Handwritten: 2000
# Caltech101_20: 6, 2386
# Caltech101: 6, 9144
# Reuters: 5, 18758
# LandUse21: 3, 2100
# Leaves: 3, 1600
# Sensit: 2, 98528
# Animal: 2, 10158