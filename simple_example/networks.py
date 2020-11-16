import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from deepfilters.systems import OdometrySystem

class OdometryData:
    def __init__(self, filename, split=8000):
        self.data = np.load(filename)

        self.split = split
        self.total = len(self.data['p_mu']) / 200
        if self.split >= self.total:
            raise ValueError("Split can't be larger than total")

        # import system
        alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
        beta = np.array([10, np.deg2rad(10)])**2
        l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21, np.nan],
                [0, 0, 0, 0, 292, 292, 292, 292, np.nan]]).T
        self.sys = OdometrySystem(alphas, beta)

        # used for predict_sigma method
        self.start_mu = self.data['start_mu']
        self.u = self.data['u']
        self.mubar = np.array( self.sys.f_np(self.start_mu.T, self.u.T) ).T
        self.start_cov = torch.tensor( self.data['start_cov'] ).float().cuda()
        self.predict_diff_mu = torch.tensor( self.mubar - self.start_mu ).float().cuda()
        self.p_cov = torch.tensor( self.data['p_cov'] ).float().cuda()

        # used for update_mu method
        self.z = self.data['z']
        self.z[np.isnan(self.z)] = 9
        self.p_mu = self.data['p_mu']
        zbar = []
        # pass landmarks and x into measurement model
        for i in range(self.z.shape[1]):
            li = l[self.z[:,i,2].astype('int')-1]
            zbar.append( self.sys.h_np(self.p_mu.T, li.T).T )
        self.zbar = np.transpose( np.array(zbar), (1,0,2))
        self.update_v = self.z[:,:,:2] - self.zbar
        # unwrap as needed
        while np.any( self.update_v[:,:,1] > np.pi ):
            self.update_v[ self.update_v[:,:,1] > np.pi ] -= 2*np.pi
        while np.any( self.update_v[:,:,1] < -np.pi ):
            self.update_v[ self.update_v[:,:,1] < -np.pi ] += 2*np.pi

        self.update_v = torch.tensor( self.update_v ).float().cuda()
        self.update_diff_mu = torch.tensor( self.data['u_mu'] - self.p_mu ).float().cuda()
        self.u_cov = torch.tensor( self.data['u_cov'] ).float().cuda()

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

    def train_predict_sigma(self, idx):
        sig_train = self.start_cov[idx*200:(idx+1)*200]
        predict_diff_mu_train = self.predict_diff_mu[idx*200:(idx+1)*200] 
        y_train = self.p_cov[idx*200:(idx+1)*200]
        return sig_train, predict_diff_mu_train, y_train

    def train_update_mu(self, idx):
        sig_train = self.p_cov[idx*200:(idx+1)*200]
        update_v = self.update_v[idx*200:(idx+1)*200]
        y_train = self.update_diff_mu[idx*200:(idx+1)*200]
        return sig_train, update_v, y_train

    def train_update_sigma(self, idx):
        x_train = self.p_cov[idx*200:(idx+1)*200]
        y_train = self.u_cov[idx*200:(idx+1)*200]
        return x_train, y_train

class Sigma(nn.Module):
    def __init__(self, sigma_size, state_size, hidden_size, n_layers):
        super().__init__()
        self.state_size = state_size
        self.sigma_size = sigma_size
        self.input_size = sigma_size**2 + state_size
        self.output_size = sigma_size**2
        self.hidden_size = hidden_size

        layers = [nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU()]
        for i in range(n_layers):
            layers.append( nn.Linear(self.hidden_size, self.hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(self.hidden_size, self.output_size) )
        
        self.net = nn.Sequential(*layers)

    def forward(self, sigma, state_predict_diff_mu):
        #do some fancy stuff to make sure it's symmetric in the end
        x = torch.cat( (sigma.view(-1, self.sigma_size**2), state_predict_diff_mu), 1)
        x = self.net( x ).reshape(-1, self.sigma_size, self.sigma_size)
        return (x + torch.transpose(x, 1, 2)) / 20
        # return x

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


if __name__ == "__main__":
    test = OdometryData("simple_example/odometry_particle_data.npz")