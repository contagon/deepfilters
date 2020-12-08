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
        self.cov_idx = torch.triu_indices(3,3)
        self.data = np.load(filename)

        self.split = split
        self.total = len(self.data['p_mu']) // 200
        if self.split >= self.total:
            raise ValueError("Split can't be larger than total")

        # import system
        alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
        beta = np.array([10, np.deg2rad(10)])**2
        l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21, np.nan],
                [0, 0, 0, 0, 292, 292, 292, 292, np.nan]]).T
        self.sys = OdometrySystem(alphas, beta)

        # used for predict_sigma networks
        self.start_mu = self.data['start_mu']
        self.u = self.data['u']
        self.mubar = np.array( self.sys.f_np(self.start_mu.T, self.u.T) ).T
        self.start_cov = torch.tensor( self.data['start_cov'] ).float().cuda()[:, self.cov_idx[0], self.cov_idx[1]]
        self.predict_diff_mu = torch.tensor( self.mubar - self.start_mu ).float().cuda()
        self.p_cov = torch.tensor( self.data['p_cov'] ).float().cuda()[:, self.cov_idx[0], self.cov_idx[1]]
        self.start_mu = torch.tensor( self.start_mu ).float().cuda()
        self.u = torch.tensor(self.u).float().cuda()

        # used for update_mu/sigma networks
        self.z = self.data['z']
        self.z[np.isnan(self.z)] = 9
        self.p_mu = self.data['p_mu']
        zbar = []
        lis = []
        # pass landmarks and x into measurement model
        for i in range(self.z.shape[1]):
            li = l[self.z[:,i,2].astype('int')-1]
            lis.append( li )
            zbar.append( self.sys.h_np(self.p_mu.T, li.T).T )
        self.lis = torch.tensor( np.transpose( np.array(lis), (1,0,2)) ).float().cuda()
        self.zbar = np.transpose( np.array(zbar), (1,0,2))
        self.update_v = self.z[:,:,:2] - self.zbar
        # unwrap as needed
        while np.any( self.update_v[:,:,1] > np.pi ):
            self.update_v[ self.update_v[:,:,1] > np.pi ] -= 2*np.pi
        while np.any( self.update_v[:,:,1] < -np.pi ):
            self.update_v[ self.update_v[:,:,1] < -np.pi ] += 2*np.pi

        self.update_v = torch.tensor( self.update_v ).float().cuda()
        self.update_diff_mu = torch.tensor( self.data['u_mu'] - self.p_mu ).float().cuda()
        self.u_cov = torch.tensor( self.data['u_cov'] ).float().cuda()[:, self.cov_idx[0], self.cov_idx[1]]
        self.z = torch.tensor( self.z[:,:,:2] ).float().cuda()

        ### NORMALIZING
        # We need to normalize all of our data to be able to use it properly
        def normalize(data):
            mask = ~torch.isnan(data).byte().any(axis=1).bool().detach()
            data[mask] = (data[mask] - data[mask].mean(axis=0)) / data[mask].std(axis=0)
            return data
        # predict sigma
        # self.predict_diff_mu = normalize( self.predict_diff_mu )
        # # update mu
        # self.p_cov = normalize( self.p_cov[:, self.cov_idx[0], self.cov_idx[1]] )
        # self.update_v = normalize( self.update_v )
        # self.update_diff_mu = normalize( self.update_diff_mu )

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
        sig_train = self.p_cov[idx*200:(idx+1)*200].detach()
        update_v = self.update_v[idx*200:(idx+1)*200]
        y_train = self.u_cov[idx*200:(idx+1)*200]
        return sig_train, update_v, y_train

    def train_all(self, idx):
        #start with loss at end of full step first...
        #then we'll get into for each step

        # for predict mu
        start_mu = self.start_mu[idx*200:(idx+1)*200]
        u = self.u[idx*200:(idx+1)*200]

        # for predict sigma
        start_sig = self.start_cov[idx*200:(idx+1)*200]

        # for update mu
        landmarks = self.lis[idx*200:(idx+1)*200]
        z = self.z[idx*200:(idx+1)*200]
        # for update sigma

        return start_mu[:-1], start_sig[:-1], u[:-1], z[:-1], landmarks[:-1], start_mu[1:], start_sig[1:] 

    def f(self, mu, u):
        x = mu[:,0] + u[:,1]*torch.cos(mu[:,2] + u[:,0])
        y = mu[:,1] + u[:,1]*torch.sin(mu[:,2] + u[:,0])
        theta = mu[:,2] + u[:,0] + u[:,2]
        return torch.stack((x,y,theta), axis=1)

    def h(self, mu, l):
        dx = l[:,0]-mu[:,0]
        dy = l[:,1]-mu[:,1]
        r = torch.sqrt( dx**2 + dy**2 )
        b = torch.atan2(dy, dx) - mu[:,2]
        return torch.stack((r,b), 1)

class Sigma(nn.Module):
    def __init__(self, sigma_size, state_size, hidden_size, n_layers):
        super().__init__()
        self.state_size = state_size
        self.sigma_size = sigma_size

        self.input_size = sigma_size*(sigma_size+1)//2 + state_size
        self.output_size = sigma_size*(sigma_size+1)//2
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
        x = torch.cat( (sigma, state_predict_diff_mu), 1)
        return self.net(x)
        # return x
    

class UpdateMu(nn.Module):
    def __init__(self, sigma_size, innovation_size, state_size, hidden_size, n_layers):
        super().__init__()
        self.sigma_size = sigma_size
        self.innovation_size = innovation_size
        self.state_size = state_size

        self.input_size = sigma_size*(sigma_size+1)//2 + innovation_size
        self.output_size = state_size
        self.hidden_size = hidden_size

        layers = [nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU()]
        for i in range(n_layers):
            layers.append( nn.Linear(self.hidden_size, self.hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(self.hidden_size, self.output_size, bias=False) )
        
        self.net = nn.Sequential(*layers)

    def forward(self, sigma, innovation):
        #do some fancy stuff to make sure it's symmetric in the end
        x = torch.cat( (sigma, innovation), 1)
        return  self.net(x)


if __name__ == "__main__":
    test = OdometryData("simple_example/odometry_particle_data.npz")