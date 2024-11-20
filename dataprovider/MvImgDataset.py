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


class MMNISTDataset(Dataset):
    """Multi-view Dataset."""

    def __init__(self, data_dir, num_views, fingerprint, transform=None):

        super().__init__()
        self.data_dir = data_dir
        self.fingerprint = fingerprint
        self.num_files = len(self.fingerprint)
        self.num_views = num_views
        self.transform = transform
        
    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        file_name = self.fingerprint[str(index)]["images"][0]
        images = [Image.open(os.path.join(self.data_dir, 'm%d'%v, file_name)) for v in range(self.num_views)]
        labels = self.fingerprint[str(index)]["labels"]
        masks = torch.tensor(self.fingerprint[str(index)]["mask"])
        permutations = torch.tensor(self.fingerprint[str(index)]["pt"])
    
        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]

        images_dict = {"m%d" % m: images[m] for m in range(self.num_views)}
        return images_dict, labels, masks, permutations, file_name  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files




def generate_dataset(data_path, 
                     num_views, 
                     missing_rate, 
                     subset=None,
                     dataset='MMNIST'):


    if dataset == 'MMNIST':
        transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize([28, 28]),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        if subset == None:
            datasets = []
            for subset in ['train', 'test']:
                print('Generating dataset_{}'.format(subset))
                subset_path = os.path.join(data_path, subset)
                fingerprint_name = 'fingerprint' + '_' + str(missing_rate) + '.pth'
                fingerprint = torch.load(os.path.join(subset_path, fingerprint_name))
 
                datasets.append(MMNISTDataset(subset_path, 5, fingerprint, transform=transform))
                print('Length of dataset_{} : {}'.format(subset, len(fingerprint)))

            return datasets
        else:
            print('Generating dataset_{}'.format(subset))
            subset_path = os.path.join(data_path, subset)
            fingerprint_name = 'fingerprint' + '_' + str(missing_rate) + '.pth'
            fingerprint = torch.load(os.path.join(subset_path, fingerprint_name))
 
            dataset = MMNISTDataset(subset_path, 5, fingerprint, transform=transform)
            print('Length of dataset_{} : {}'.format(subset, len(fingerprint)))

            return dataset
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_views', type=int, default=5)
    parser.add_argument('--data_path', type=str, default='/public/home/gaoxin/MVP/data/MMNIST')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    args = parser.parse_args()  # use vars to convert args into a dict
    print("\nARGS:\n", args)

    # generate dataset
    dataset_train = generate_dataset(args.data_path, args.num_views, 
                                                args.missing_rate, 'train')

    # check dataset
    fig, axes = plt.subplots(4, 5, figsize=(5 * 4, 4 * 4))
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(4):
        images_dict, label, masks, permutations, _ = dataset_train[(i+1)*130]
        for j in range(5):
            img_tensor = images_dict["m%d" % j]   
            # Move the color channel to the last dimension
            img = img_tensor.permute(1, 2, 0).numpy()     
            # Normalize img to range [0, 1] for displaying
            img = (img + 1) / 2
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
    
    plt.savefig('2.jpg', bbox_inches='tight')
    plt.show()
    
    print("Label:", label)
    print("Mask:", masks)
    print("permutaitons:", permutations.shape)  