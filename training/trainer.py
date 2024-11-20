## trainer for MvDataset
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
import os
import json
from datetime import datetime
from time import time, sleep
import random
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.data
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join, isfile, maybe_mkdir_p
import sys
sys.path.append('/path/to/MVP')
from utils import get_logger, empty_cache, collate_outputs
from model.vanilla_mvVAE import VanillaMVVAE   # 记得改模型
from model.ICMVLoss import ICMVLoss 
from dataprovider.MvDataset import generate_dataset
from torch.cuda import device_count
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
# from training.early_stopping import EarlyStopping
import math
import csv


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

class ICMVtrainer(object):
    def __init__(self, args, device: torch.device = torch.device('cuda')):
        
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        # self.log_path = os.path.join(args.log_path, "log_SensIT_Vehicle_missing_0.1_2024-08-05_21-31-55")
        self.log_path = os.path.join(args.log_path, f"log_{args.dataset}_{args.missing_rate}_{timestamp}")
        # 创建必要的子文件夹
        self.plt_path = os.path.join(self.log_path, 't-SNE')
        self.recon_path = os.path.join(self.log_path, 'recon')
        self.save_path = os.path.join(self.log_path, 'checkpoint')
        # 创建所有需要的目录
        for path in [self.log_path, self.plt_path, self.recon_path, self.save_path]:
            os.makedirs(path, exist_ok=True)
            
        self.num_views = args.num_views
        self.data_path = args.data_path
        self.missing_rate = args.missing_rate
        self.in_channels = args.in_channels
        self.num_classes = args.num_classes
        self.arch = args.arch
        self.transform = args.transform
        self.latent_dim = args.latent_dim
        self.class_dim = args.class_dim
        self.batchsize = args.batch_size
        self.device = device
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.epochs
        self.interval = args.interval
        self.normalizing_factor = args.normalizing_factor
        self.weight = args.weight
        self.likelihood = args.likelihood
        self.seed_list = args.seed
        self.current_epoch = 1

        self.logger = get_logger(self.log_path, "train_log")
        formatted_params = json.dumps(self.__dict__, indent=4, default=str)
        self.logger.info("Training parameters: %s", formatted_params)
        self.writer = SummaryWriter(log_dir=self.log_path)
        # 在 `ICMVtrainer` 的 `__init__` 方法中添加用于存储损失记录的文件路径：
        self.csv_path = os.path.join(self.log_path, "loss_records.csv")
        # 创建CSV文件并写入标题
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Experiment", "Epoch", "Loss_Type", "Loss_Value"])

        self.was_initialized = False

    def initialize(self):

        if not self.was_initialized:
            self.network = self.build_network_architecture().to(self.device)
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [200], gamma=0.5, last_epoch=-1)
            self.loss = ICMVLoss(num_views=self.num_views,
                                 latent_dim=self.latent_dim,
                                 num_classes=self.num_classes,
                                 normalizing_factor=self.normalizing_factor,
                                 weight = self.weight,
                                 likelihood = self.likelihood,
                                 device=self.device)
            self.was_initialized = True


    def build_network_architecture(self) -> torch.nn.Module: 

        model = VanillaMVVAE(num_views=self.num_views, num_classes=self.num_classes, latent_dim=self.latent_dim, class_dim=self.class_dim,
                             device=self.device, in_channels=self.in_channels, architecture=self.arch, transform=self.transform)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        return model
    

    def get_dataloaders(self):
        
        dataset_str = self.data_path.split('/')[-1]

        dataset_train, dataset_test = generate_dataset(dataset_str, self.data_path, self.missing_rate)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batchsize, num_workers=4, shuffle=True,)
        print("Trainloader length:", len(train_loader))
        if dataset_test != None:
            test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=False)
            print("Testloader length:", len(test_loader))
        else:
            test_loader = None

        return train_loader, test_loader


    def on_train_start(self, load_pretrain=False):

        seed_torch(self.seed)
        if not self.was_initialized:
            self.initialize()  
        empty_cache(self.device)
        self.dataloader_train, self.dataloader_test = self.get_dataloaders()
        self.current_epoch = 1
        if load_pretrain:
            self.logger.info('load state dict from pretrain......')
            checkpoint_path = os.path.join(self.save_path, 'pretrain', 'checkpoint_final.pth')
            checkpoint = torch.load(checkpoint_path)
            self.network.load_state_dict(checkpoint['network_weights'])
            
    def on_train_end(self):

        self.current_epoch -= 1
        self.save_checkpoint(join(self.save_path_now, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if isfile(join(self.save_path_now, "checkpoint_latest.pth")):
            os.remove(join(self.save_path_now, "checkpoint_latest.pth"))

        empty_cache(self.device)
        self.was_initialized = False
        self.logger.info("Training done.")


    def train_step(self, batch: dict, id: int, double: bool) -> dict:
        self.network.train()
        data_dict, label, masks, permutations = batch
        data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
        label = label.to(self.device)
        masks = masks.to(self.device)
        permutations = permutations.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        recon_dict, mus, logvars, clsmu, clslogvar = self.network(data_dict, masks, permutations, double)
        masks_ = masks.unsqueeze(-1).repeat(1, 1, clsmu[0].shape[-1])
        exist_mu = masks_ * clsmu[0]
        T = 1./torch.exp(clslogvar[0])
        exist_T = masks_ * T
        aggregated_T = torch.sum(exist_T, dim=1)
        aggregated_mu_numerator = exist_mu * exist_T
        aggregated_mu = torch.sum(aggregated_mu_numerator, dim=1) / aggregated_T
        average = aggregated_mu.detach()
        loss_dict = self.loss(data_dict=[data_dict, recon_dict], mus=mus, logvars=logvars, 
                                    clsmus=clsmu, clslogvars=clslogvar, masks=masks)
        
        loss = loss_dict['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        for k, v in loss_dict.items():
            loss_dict[k] = (v).detach().cpu().numpy()
            print('{} : {:.4f}'.format(k, v)) 

        return loss_dict, average, label

    # def collect_loss(self, train_outputs: List[dict], tensorboard=True):
    #     outputs = collate_outputs(train_outputs)

    #     if tensorboard:
    #         for k, v in outputs.items():
    #             loss_here = np.mean(v) #/len(self.dataloader_train)
    #             self.writer.add_scalar('loss/'+k, loss_here, self.current_epoch)

    #     self.logger.info('train_losses:{:.4f}'.format(np.mean(outputs['loss'])))

    def collect_loss(self, train_outputs: List[dict], experiment_num: int, tensorboard=True):
        outputs = collate_outputs(train_outputs)

        if tensorboard:
            for k, v in outputs.items():
                loss_here = np.mean(v)
                self.writer.add_scalar('loss/'+k, loss_here, self.current_epoch)

        self.logger.info('train_losses:{:.4f}'.format(np.mean(outputs['loss'])))

        # 将损失值写入CSV
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for k, v in outputs.items():
                loss_here = np.mean(v)
                writer.writerow([experiment_num, self.current_epoch, k, loss_here])



    def Clustering(self, averages, labels, tensorboard=True):
        average = torch.cat(averages)
        label = torch.cat(labels)

        def acc(y_true, y_pred):
            """
            Calculate clustering accuracy. Require scikit-learn installed
            # Arguments
                y: true labels, numpy.array with shape `(n_samples,)`
                y_pred: predicted labels, numpy.array with shape `(n_samples,)`
            # Return
                accuracy, in [0,1]
            """
            y_true = y_true.astype(np.int64)
            assert y_pred.size == y_true.size
            D = max(y_pred.max(), y_true.max()) + 1
            w = np.zeros((D, D), dtype=np.int64)
            for i in range(y_pred.size):
                w[y_pred[i], y_true[i]] += 1
            ind = linear_sum_assignment(w.max() - w)
            ind = np.array(ind).T
            return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        
        kmeans = KMeans(n_clusters = self.num_classes, random_state=self.seed, n_init='auto')
        kmeans.fit(average.detach().cpu())

        ClAccuracy = acc(label.cpu().numpy(), kmeans.labels_)
        NMI = metrics.normalized_mutual_info_score(label.cpu().numpy(), kmeans.labels_)
        ARI = metrics.adjusted_rand_score(label.cpu().numpy(), kmeans.labels_)
        if tensorboard:
            self.writer.add_scalar('metric/acc', ClAccuracy, self.current_epoch)
            self.writer.add_scalar('metric/NMI', NMI, self.current_epoch)
            self.writer.add_scalar('metric/ARI', ARI, self.current_epoch)
        self.logger.info('Clustering model accuracy = {:.4f}'.format(ClAccuracy))
        self.logger.info('NMI = {:.4f}'.format(NMI))
        self.logger.info('ARI = {:.4f}'.format(ARI))
        
        return ClAccuracy, NMI, ARI

    def save_checkpoint(self, filename: str) -> None:

        model = self.network

        checkpoint = {
            'network_weights': model.state_dict(),
            'current_epoch': self.current_epoch,
        }
        torch.save(checkpoint, filename)
      
    
    def run_training(self):

        ############################### training for five times ################################
        perform_num = int(self.num_epochs//self.interval)
        performances = [np.zeros(shape=(len(self.seed_list), 3)) for _ in range(perform_num)]
        for k, seed_ in enumerate(self.seed_list):
            print("---------------------------run_times------------------------------------")
            print(k)
            print("------------------------------------------------------------------------")
            self.save_path_now = os.path.join(self.save_path, str(k))
            os.makedirs(self.save_path_now, exist_ok=True)
            self.seed = seed_
            self.on_train_start(load_pretrain=False)
            for epoch in range(1, self.num_epochs+1):
                warmup_done = False if epoch <= 100 else True
                self.logger.info('============Train_epoch:{}============'.format(epoch))
                self.logger.info(
                            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
                train_outputs, averages, labels = [], [], []
                for batch_id, batch in tqdm(enumerate(self.dataloader_train)):
                    train_output, average, label= self.train_step(batch, batch_id, double=warmup_done)
                    train_outputs.append(train_output)
                    averages.append(average)
                    labels.append(label)
                # self.collect_loss(train_outputs)
                self.collect_loss(train_outputs, experiment_num=k) # 调用 collect_loss 时传入当前实验编号
                ACC, NMI, ARI = self.Clustering(averages, labels)
                if epoch > 1 and (epoch) % self.interval == 0:
                    performances[epoch//self.interval-1][k,0] = ACC
                    performances[epoch//self.interval-1][k,1] = NMI
                    performances[epoch//self.interval-1][k,2] = ARI
                    self.save_checkpoint(join(self.save_path_now, "checkpoint_%d.pth"%epoch))
                self.lr_scheduler.step()
                # self.save_checkpoint(join(self.save_path_now, "checkpoint_latest.pth"))
                self.current_epoch += 1
            self.on_train_end()
        
        ############################## logging ################################
        for i in range(1, perform_num+1):
            self.logger.info('epoch: %d' % (i*self.interval))
            self.logger.info('--------------------------------------')
            means_per = np.around(np.mean(performances[i-1], axis=0), 4)
            std_per = np.around(np.std(performances[i-1], axis=0), 4)
            self.logger.info(f'{list(means_per)}')
            self.logger.info(f'{list(std_per)}')

