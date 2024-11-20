import numpy as np
import argparse
import torch
import os
import glob
import json
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy
import torch
from complete_shuffle_package import *
from tqdm import tqdm


class MvDataset(Dataset):
    """Multi-view Dataset."""

    def __init__(self, multi_view_data, fingerprint, transform=None):

        super().__init__()
        self.multi_view_data = multi_view_data
        self.fingerprint = fingerprint
        self.num_files = len(self.fingerprint)
        self.num_views = len(multi_view_data)
        self.transform = transform
        
    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        datas = [torch.tensor(self.multi_view_data[m].loc[index,:]).float() for m in range(self.num_views)]
        masks = torch.tensor(self.fingerprint[str(index)]["mask"])
        permutations = torch.tensor(self.fingerprint[str(index)]["pt"])
        labels = torch.tensor(self.fingerprint[str(index)]["labels"])

        # transforms
        if self.transform:
            datas = [self.transform(d) for d in datas]

        datas_dict = {"m%d" % m: datas[m] for m in range(self.num_views)}
        return datas_dict, labels, masks, permutations  # NOTE: labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files



def generate_dataset(dataset_str, 
                     data_path, 
                     missing_rate, 
                     transform=None):
    
    if dataset_str == 'Handwritten':
        view_names = ['mfeat-fou', 'mfeat-fac', 'mfeat-kar', 'mfeat-zer', 'mfeat-pix', 'mfeat-mor']   
    elif dataset_str == 'Caltech101_20' or dataset_str == 'Caltech101':
        view_names = ['0_Gabor48.csv', '1_WM40.csv', '2_cenhist254.csv', '3_HOG1984.csv', '4_GIST512.csv', '5_LBP928.csv']
    elif dataset_str == 'Reuters10':
        view_names = ['1_English.csv', '2_French.csv', '3_German.csv', '4_Italian.csv', '5_Spanish.csv']
    elif dataset_str == 'Scene15' or dataset_str == 'LandUse21':
        view_names = ['0_GIST20.csv', '1_PHOG59.csv', '2_LBP40.csv']
    elif dataset_str == 'SensIT_Vehicle':
        view_names = ['0_acoustic.csv', '1_seismic.csv']
    elif dataset_str == 'Animal':
        view_names = ['0_4096.csv', '1_4096.csv']
    elif dataset_str == 'CUB':
        view_names = ['0_1024.csv', '1_300.csv']
    elif dataset_str == 'leaves100':
        view_names = ['0_margin64.csv', '1_shape64.csv', '2_texture64.csv']

    fingerprint_name = 'fingerprint' + '_' + str(missing_rate) + '.pth'
    fingerprint = torch.load(os.path.join(data_path, fingerprint_name))

    num_views = len(view_names)
    multi_view_data = []
    if dataset_str == 'Handwritten':
        for i in range(num_views):
            data = np.loadtxt(os.path.join(data_path, view_names[i]), dtype=float)
            df = pd.DataFrame(data)
            df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
            multi_view_data.append(df_norm)
    elif dataset_str == 'Animal':
        for i in range(num_views):
            df = pd.read_csv(os.path.join(data_path, view_names[i]), encoding="utf-8", header=None)
            df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x)-np.min(x)))
            multi_view_data.append(df_norm)
    else:
        for i in range(num_views):
            df = pd.read_csv(os.path.join(data_path, view_names[i]), encoding="utf-8", header=None)
            df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
            multi_view_data.append(df_norm)   

    dataset_train = MvDataset(multi_view_data, fingerprint, transform=transform)
    dataset_test = None

    return dataset_train, dataset_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_views', type=int, default=6)
    parser.add_argument('--data_path', type=str, default='/path/to/Handwritten')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    args = parser.parse_args()  
    print("\nARGS:\n", args)

    # generate dataset
    dataset_str = args.data_path.split('/')[-1]
    print(dataset_str)
 
    dataset_train, dataset_test = generate_dataset(dataset_str, args.data_path, args.missing_rate)

    # check dataset
    print(len(dataset_train))
    images_dict, label, masks, permutations = dataset_train[15]
    for m in range(args.num_views):
        print("Modality %d:" % m, images_dict["m%d" % m].shape)
    
    print("Label:", label)
    print("Mask:", masks)
    print("permutaitons:", permutations.shape)  