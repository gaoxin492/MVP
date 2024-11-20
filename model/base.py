from torch import tensor as Tensor
from typing import List, Any, Tuple
from torch import nn
import torch
from abc import abstractmethod
import math
from torch.nn import functional as F
import numpy as np

eta = 1e-20

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



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)



class Encoder1(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.shared_encoder = nn.Sequential(                          # input shape (3, 64, 64)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),     # -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),    # -> (128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),   # -> (256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),   # -> (512, 4, 4)
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),   # -> (512, 2, 2)
            nn.ReLU(),
            Flatten(),  
            nn.Linear(2048, latent_dim),       # -> (latent_dim)
            nn.ReLU(),
        )

        self.class_mu = nn.Linear(latent_dim, latent_dim)
        self.class_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, x):
        h = self.shared_encoder(x)
        return self.class_mu(h), self.class_logvar(h)


class Decoder1(nn.Module):
    def __init__(self, latent_dim, class_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+class_dim, 2048),                                                        # -> (1024)
            nn.ReLU(),
            Unflatten((512, 2, 2)),     
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (512, 4, 4)
            nn.ReLU(),                                                     
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (256, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 64, 64)
            nn.Tanh()
        )

    def forward(self, c, z):
        x = torch.cat([c,z], dim=-1)

        return self.decoder(x)



class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),     # -> (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),    # -> (128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),   # -> (256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),   # -> (512, 2, 2)
            nn.ReLU(),
            Flatten(),                                                # -> (1024)
            nn.Linear(2048, latent_dim),       # -> (latent_dim)
            nn.ReLU(),
        )

        self.class_mu = nn.Linear(latent_dim, latent_dim)
        self.class_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        return self.class_mu(h), self.class_logvar(h)




class Decoder2(nn.Module):
    def __init__(self, latent_dim, class_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+class_dim, 2048),                                                        # -> (1024)
            nn.ReLU(),
            Unflatten((512, 2, 2)),                                                            # -> (512, 2, 2)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (256, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),                   # -> (128, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
            nn.Tanh()
        )

    def forward(self, c, z):
        x = torch.cat([c,z], dim=-1)

        return self.decoder(x)




class BaseMVVAE(nn.Module):
        
        def __init__(self) -> None:
            super(BaseMVVAE, self).__init__()
    
        def generate(self, x: Tensor, **kwargs) -> Tensor:
            raise NotImplementedError
        
        def combination(self, *inputs: Tensor) -> Tensor:
            raise NotImplementedError
    
        @abstractmethod
        def forward(self, *inputs: Any) -> Tensor:
            pass
    
        @abstractmethod
        def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
            pass



class encoder_mlp(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 arch: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(encoder_mlp, self).__init__()

        self.latent_dim = latent_dim
    
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, arch[1]),
            nn.ReLU(),
            nn.Linear(arch[1], arch[2]),
            nn.ReLU(),
            nn.Linear(arch[2], arch[3]),
            nn.ReLU(),
        )

        self.fc_mu = nn.Sequential(nn.Linear(arch[3], latent_dim),
                                    nn.Tanh())
        self.fc_var = nn.Sequential(nn.Linear(arch[3], latent_dim),
                                    nn.Tanh())
  

    def forward(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # cls = self.fc_cls(result)

        return [mu, log_var] #, cls]
    
class decoder_mlp(nn.Module):
    def __init__(self, 
                 out_dim: int, 
                 arch: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(decoder_mlp, self).__init__()
        
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(latent_dim, arch[3]),
                                nn.ReLU())

        self.decoder = nn.Sequential(
            nn.Linear(arch[3], arch[2]),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.Linear(arch[2], arch[1]),
            nn.ReLU(),
            nn.Linear(arch[1], out_dim),
            nn.Tanh()
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        deconv_input = self.fc(z)
        result = self.decoder(deconv_input)
        return result