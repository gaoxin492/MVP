import torch
from model.base import (BaseMVVAE, Encoder2, Decoder2, 
                            Encoder1, Decoder1)
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from typing import List, Any
import random
import math
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class Unflatten(nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, arch):
        super(MLP, self).__init__()

        layers = []
        current_dim = in_dim
        for i in range(len(arch)):
            layers.append(nn.Linear(current_dim, arch[i]))
            layers.append(nn.LeakyReLU())
            current_dim = arch[i]
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):

        return self.net(x)
    
    
class VanillaMVVAE(BaseMVVAE):

    def __init__(self,
               dataset: str,
               num_views: int,
               num_classes: int,
               latent_dim: int,
               class_dim: int,
               **kwargs) -> None:
        super(VanillaMVVAE, self).__init__()
        
        self.num_views = num_views
        self.arch = kwargs['arch']
        self.transform = kwargs['tansformation']
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.device = kwargs['device']
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        if dataset == 'MVShapeNet':
            for v in range(num_views):
                self.encoders["encoder%d" % v] = Encoder1(latent_dim) 
                self.decoders["decoder%d" % v] = Decoder1(latent_dim, class_dim) 
        elif dataset == 'MMNIST':
            for v in range(num_views):
                self.encoders["encoder%d" % v] = Encoder2(latent_dim)
                self.decoders["decoder%d" % v] = Decoder2(latent_dim, class_dim)

        self.transforms_mu = nn.ModuleDict()
        self.transforms_logvar = nn.ModuleDict()
        for i in range(num_views):
            for j in range(num_views):
                if i != j:
                    self.transforms_mu["transform%d%d" % (i, j)] = MLP(latent_dim, latent_dim, self.transform)
                    self.transforms_logvar["transform%d%d" % (i, j)] = MLP(latent_dim, latent_dim, self.transform)


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def combination(self, mus: Tensor, logvars: Tensor) -> List[Tensor]:
        """
        Combination function to combine the latent variables
        :param inputs: (Tensor) [B x K x D]
        :return: (Tensor) [B x D]
        """
        if len(mus.shape)==4:
            d=2
        else:
            d=1
        N = mus.shape[d]
        vars = torch.exp(logvars)
        sigma_z_star = 1/torch.sum(1/vars, dim=d)
        mu_z_star = sigma_z_star * torch.sum(mus/vars, dim=d)
        return [mu_z_star, torch.log(sigma_z_star)]

    
    def combination_masked(self, mus: Tensor, logvars: Tensor, masks: Tensor) -> List[Tensor]:

        masks = masks.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
        exist_mu = masks * mus
        T = 1./torch.exp(logvars)
        exist_T = masks * T
        aggregated_T = torch.sum(exist_T, dim=1)
        aggregated_var = 1. / aggregated_T
        aggregated_mu_numerator = exist_mu * exist_T
        aggregated_mu = torch.sum(aggregated_mu_numerator, dim=1) / aggregated_T

        return [aggregated_mu, torch.log(aggregated_var)]
        

    def complete_partition(self, mus, logvars, permutations):

        B = permutations.shape[0]
        # Random Complete partition
        mus0, logvars0 = mus.clone(), logvars.clone()
        mus, logvars = mus.permute(3, 0, 1, 2), logvars.permute(3, 0, 1, 2)
        mus1 = torch.zeros(self.latent_dim, B, self.num_views, self.num_views).to(self.device)
        logvars1 = torch.zeros(self.latent_dim, B, self.num_views, self.num_views).to(self.device)   
        idx = list(range(permutations.shape[1]))
        random.shuffle(idx)
        for v in range(self.num_views):
            k = idx[v]
            mus1[:, :, :, v] = mus[:, :, :, v].gather(dim=2, index=permutations[:,k,:].repeat(self.latent_dim, 1, 1))
            logvars1[:, :, :, v] = logvars[:, :, :, v].gather(dim=2, index=permutations[:,k,:].repeat(self.latent_dim, 1, 1))
        mus1, logvars1 = mus1.permute(1, 2, 3, 0), logvars1.permute(1, 2, 3, 0)

        return mus1, logvars1 
        

    def forward(self, image_dict: dict, masks: Tensor, permutations: Tensor, double=False) -> List[Any]:

        B = masks.shape[0]
        # Two types of latent variables, use matrix for later permutation
        mus = torch.zeros(B, self.num_views, self.num_views, self.latent_dim).to(self.device)
        logvars = torch.zeros(B, self.num_views, self.num_views, self.latent_dim).to(self.device)
        # Encode and transform
        for v in range(self.num_views):
            mu, logvar = self.encoders["encoder%d" % v](image_dict["m%d" % v])
            mus[:,v,v,:] = mu
            logvars[:,v,v,:] = logvar
            for j in range(self.num_views):
                if j != v:
                    mus[:,v,j,:] = self.transforms_mu["transform%d%d" % (v, j)](mu)
                    logvars[:,v,j,:] = self.transforms_logvar["transform%d%d" % (v, j)](logvar)

        mus1, logvars1 = self.complete_partition(mus, logvars, permutations)
        recon_dict0 = {}
        recon_dict1 = {} if double else None   
        # Decode
        clsmus0, clslogvar0, clsmus1, clslogvar1 = [],[],[],[]
        for v in range(self.num_views):
            mu_combined, logvar_combined = self.combination(mus[:, v, :, :self.class_dim], logvars[:, v, :, :self.class_dim])
            clsmus0.append(mu_combined)
            clslogvar0.append(logvar_combined)
            
            c = self.reparameterize(mu_combined, logvar_combined)
            z = self.reparameterize(mus[:, v, v, :], logvars[:, v, v, :]) 
            recon_dict0["m%d" % v] = self.decoders["decoder%d" % v](c, z) #self.decode_c(c)
            
            mu_combined, logvar_combined = self.combination(mus1[:, v, :, :self.class_dim], logvars1[:, v, :, :self.class_dim])
            clsmus1.append(mu_combined)
            clslogvar1.append(logvar_combined)
            
            if double:
                c = self.reparameterize(mu_combined, logvar_combined)
                z = self.reparameterize(mus1[:, v, v, :], logvars1[:, v, v, :])
                recon_dict1["m%d" % v] = self.decoders["decoder%d" % v](c, z)

        clsmus0 = torch.stack(clsmus0, dim=1)  # 将dim=1作为stack维度
        clslogvar0 = torch.stack(clslogvar0, dim=1)
        clsmus1 = torch.stack(clsmus1, dim=1)
        clslogvar1 = torch.stack(clslogvar1, dim=1)

        return [recon_dict0, recon_dict1], [mus, mus1], [logvars, logvars1], [clsmus0, clsmus1], [clslogvar0, clslogvar1]
            

    
    
