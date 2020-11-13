import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class OdometryData:
    def __init__(self, filename, split=8000):
        self.data = np.load(filename)
        self.p_mu = torch.tensor( self.data['p_mu'] ).float().cuda()
        self.p_cov = torch.tensor( self.data['p_cov'] ).float().cuda()
        self.u_mu = torch.tensor( self.data['u_mu'] ).float().cuda()
        self.u_cov = torch.tensor( self.data['u_cov'] ).float().cuda()
        self.u = torch.tensor( self.data['u'] ).float().cuda()
        self.split = split
        self.total = len(self.p_mu) / 200

    @property
    def rand_train_idx(self):
        idx = np.arange(0, self.split)
        np.random.shuffle(idx)
        return idx

    @property
    def rand_test_idx(self):
        idx = np.arange(self.split, self.total)
        np.random.shuffle(idx)
        return idx

    def train_predict_mu(self, idx):
        x_train = self.u_mu[idx*200:(idx+1)*200-1]
        y_train = self.p_mu[idx*200+1:(idx+1)*200]
        return x_train, y_train

    def train_predict_sigma(self, idx):
        x_train = self.u_cov[idx*200:(idx+1)*200-1]
        y_train = self.p_cov[idx*200+1:(idx+1)*200]
        return x_train, y_train

    def train_update_mu(self, idx):
        x_train = self.p_mu[idx*200:(idx+1)*200]
        y_train = self.u_mu[idx*200:(idx+1)*200]
        return x_train, y_train

    def train_update_sigma(self, idx):
        x_train = self.p_cov[idx*200:(idx+1)*200]
        y_train = self.u_cov[idx*200:(idx+1)*200]
        return x_train, y_train

class Sigma(nn.Module):
    def __init__(self, sigma_size, hidden_size, n_layers):
        super().__init__()
        self.sigma_size = sigma_size
        self.input_size = sigma_size**2
        self.output_size = sigma_size**2
        self.hidden_size = hidden_size

        layers = [nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU()]
        for i in range(n_layers):
            layers.append( nn.Linear(self.hidden_size, self.hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(self.hidden_size, self.output_size) )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #do some fancy stuff to make sure it's symmetric in the end
        x = self.net( x.view(-1, self.sigma_size**2) ).reshape(-1, self.sigma_size, self.sigma_size)
        # return (x + torch.transpose(x, 1, 2)) / 2
        return x

class UpdateMu(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        layers = [nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU()]
        for i in range(n_layers):
            layers.append( nn.Linear(self.hidden_size, self.hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(self.hidden_size, self.output_size) )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #do some fancy stuff to make sure it's symmetric in the end
        return  self.net(x)