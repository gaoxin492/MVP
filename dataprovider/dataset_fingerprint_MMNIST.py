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

def generate_fingerprint(data_path, num_modalities, missing_rate, num, seed=42):

    for mode in ['train', 'test']:
        print(mode)
        data_path_m = os.path.join(data_path, mode)

        modalities_path = [os.path.join(data_path_m, 'm%d' % m) for m in range(num_modalities)]

        file_paths = {dp: [] for dp in modalities_path}
        
        for dp in modalities_path:
            files = glob.glob(os.path.join(dp, '*.png'))
            file_paths[dp] = files

        fingerprint_json = {i: {} for i in range(len(file_paths[modalities_path[0]]))}
        random.seed(seed)
        torch.manual_seed(seed)

        num_samples = [len(file_path) for file_path in file_paths.values()]
        assert num_samples[0]==num_samples[1]==num_samples[2]==num_samples[3]==num_samples[4]
        index = [i for i in range(num_samples[0])]
        random.shuffle(index)
        index_miss = index[:int(num_samples[0]*missing_rate)]
        index_preserve = index[int(num_samples[0]*missing_rate):]
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
            fingerprint_json[mv_samples]['images'] = [file_paths[dp][mv_samples] for dp in modalities_path]
            fingerprint_json[mv_samples]['labels'] = [int(file_paths[dp][mv_samples].split('.')[-2]) for dp in modalities_path]

        masks = torch.ones(num_modalities)
        all_permutations = total_permutation(masks.tolist(), num)
        for mv_samples in tqdm(index_preserve):
            fingerprint_json[mv_samples]['mask'] = masks.tolist()
            fingerprint_json[mv_samples]['pt'] = all_permutations
            fingerprint_json[mv_samples]['images'] = [file_paths[dp][mv_samples] for dp in modalities_path]
            fingerprint_json[mv_samples]['labels'] = [int(file_paths[dp][mv_samples].split('.')[-2]) for dp in modalities_path]


        json_name = 'fingerprint' + str(num_views) + '_' + str(int(10*missing_rate)) + '.json'
        save_path = os.path.join(data_path_m, json_name)
        with open(save_path, 'w') as f:
            json.dump(fingerprint_json, f, indent=4)
    
num_views = 5
data_path = '/path/to/MMNIST'
num_pt = max(num_views, math.factorial(num_views-1))

for missing_rate in [0.1, 0.3, 0.5, 0.7]:
    generate_fingerprint(data_path, num_views, missing_rate, num=num_pt, seed=42)
