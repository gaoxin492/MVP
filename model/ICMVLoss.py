import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor as Tensor
from typing import List, Any
import random
import math
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def combination_masked(mus: Tensor, logvars: Tensor, masks: Tensor) -> List[Tensor]:

    if len(mus.shape)==4:
        d=2
        masks = masks.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
        masks = masks.unsqueeze(1).repeat(1, mus.shape[1], 1, 1)
    else:
        d=1
        masks = masks.unsqueeze(-1).repeat(1, 1, mus.shape[-1])
    exist_mu = masks * mus
    T = 1./torch.exp(logvars)
    exist_T = masks * T
    aggregated_T = torch.sum(exist_T, dim=d)
    aggregated_var = 1. / aggregated_T
    aggregated_mu_numerator = exist_mu * exist_T
    aggregated_mu = torch.sum(aggregated_mu_numerator, dim=d) / aggregated_T

    return [aggregated_mu, torch.log(aggregated_var)]


class ICMVLoss(nn.Module):
    def __init__(self, num_views, latent_dim, num_classes, 
                    normalizing_factor, weight, likelihood, device):
        super(ICMVLoss, self).__init__()

        self.num_views = num_views
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        self.normalizing_factor = normalizing_factor
        self.w1 = weight[0]
        self.w2 = weight[1]
        self.likelihood = likelihood

    def KL_divergence(self, mu0: Tensor, logvar0: Tensor, mu1: Tensor, logvar1: Tensor) -> Tensor:
        """
        KL divergence losses between two Gaussian distributions (diagonal covariance matrix)
        """
        KL = 0.5*(logvar1 - logvar0) + (torch.exp(logvar0) + (mu0 - mu1)**2) / (2 * torch.exp(logvar1)) - 0.5 

        return torch.sum(KL, dim=-1)


    def forward(self, 
                data_dict: List[dict],
                masks: Tensor,
                labels=None,
                **kwargs) -> dict:


        loss = 0
        # MSE of reconstruction
        img, recon_list = data_dict[0], data_dict[1]
        NLL = torch.zeros(masks.shape[0], self.num_views).to(self.device)
        div = 0
        for recon in recon_list:
            if recon is not None:
                for k,v in recon.items():
                    if self.likelihood == 'Laplace':
                        NLL[:, int(k[-1])] += torch.sum(F.l1_loss(torch.flatten(v, start_dim=1), 
                                            torch.flatten(img[k],start_dim=1), reduction='none'), dim=-1)
                    else:
                        NLL[:, int(k[-1])] += torch.sum(F.mse_loss(torch.flatten(v, start_dim=1), 
                                                torch.flatten(img[k],start_dim=1), reduction='none'), dim=-1)
                div += 1
        NLL = NLL / div * masks
        NLL = torch.sum(NLL)/NLL.shape[0]
        loss += self.normalizing_factor[0]*NLL
        loss_dict = {'Reconstruction_Loss': self.normalizing_factor[0]*NLL}

        # KL divergence of view-peculiar latent variable
        if 'mus' in kwargs.keys():
            mus = kwargs['mus']
            logvars = kwargs['logvars']
            mu0, mu1 = mus[0].view(-1, self.latent_dim), mus[1].view(-1, self.latent_dim)
            logvar0, logvar1 = logvars[0].view(-1, self.latent_dim), logvars[1].view(-1, self.latent_dim)
            KLD = self.KL_divergence(mu0, logvar0, mu1, logvar1) / self.num_views
            if recon_list[1] is not None:
                KLD += self.KL_divergence(mu1, logvar1, mu0, logvar0) / self.num_views
                KLD /= 2
            KLD = KLD.view(masks.shape[0], self.num_views, self.num_views) 
            KLD = KLD * masks.unsqueeze(-1).repeat(1, 1, self.num_views)
            KLD = torch.sum(KLD, dim=2) * ((masks+1)/2)
            KLD = torch.sum(KLD, dim=-1).mean()
            loss += self.w1*self.normalizing_factor[1]*KLD
            loss_dict.update({'KLD': self.w1*self.normalizing_factor[1]*KLD})

        # KL divergence of common representation
        if 'clsmus' in kwargs.keys():
            clsmus = kwargs['clsmus']
            clslogvars = kwargs['clslogvars']
            d = clsmus[0].shape[-1]
            clsmu0, clsmu1 = clsmus[0].view(-1, d), clsmus[1].view(-1, d)
            clslogvar0, clslogvar1 = clslogvars[0].view(-1, d), clslogvars[1].view(-1, d)
            KLD_o = self.KL_divergence(clsmu0, clslogvar0, clsmu1, clslogvar1)
            if recon_list[1] is not None:
                KLD_o += self.KL_divergence(clsmu1, clslogvar1, clsmu0, clslogvar0)
                KLD_o /= 2
            KLD_o = KLD_o.view(masks.shape[0], self.num_views)
            KLD_o = KLD_o * masks
            KLD_o = torch.sum(KLD_o, dim=1).mean()
            loss += self.w2*self.normalizing_factor[1]*KLD_o
            loss_dict.update({'KLD_o': self.w2*self.normalizing_factor[1]*KLD_o})

        loss_dict.update({'loss': loss})
        return loss_dict










            
        


