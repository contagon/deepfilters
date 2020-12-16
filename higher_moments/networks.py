import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from deepfilters.systems import OdometrySystem
import pickle
        
class Normalize(nn.Module):
    def __init__(self, data):
        super().__init__()
        data = data.clone().reshape(-1, data.shape[-1])
        mask = ~torch.isnan(data).byte().any(axis=1).bool().detach()
        self.mean = data[mask].mean(axis=0)
        self.std  = data[mask].std(axis=0)
        self.std[self.std < .1] = .1

    def forward(self, x):
        return (x - self.mean) / self.std

    def undo(self, x):
        return x * self.std + self.mean

class OdometryData:
    def __init__(self, filename, split=8000):
        self.cov_idx = torch.triu_indices(3,3)
        data = pickle.load( open(filename, 'rb') )

        self.split = split
        self.total = data['start'][1].shape[0]
        if self.split >= self.total:
            raise ValueError("Split can't be larger than total")

        # import system
        l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21, np.nan],
                [0, 0, 0, 0, 292, 292, 292, 292, np.nan]]).T

        # push mu through system with controls
        start_mu = data['start'][1]
        u        = data['us']
        predicted_mu = np.zeros_like(start_mu)
        for i in range(start_mu.shape[0]):
            predicted_mu[i] = self.f_np(start_mu[i], u[i])
        predict_mu_diff = predicted_mu - start_mu

        # calculate innovation for all data
        z = data['zs']
        z[np.isnan(z)] = 9
        middle_mu = data['middle'][1]
        zbar = np.zeros((z.shape[0], z.shape[1], z.shape[2], 2))
        lis = l[z[:,:,:,2].astype('int')-1]
        # pass landmarks and state into measurement model
        for i in range(z.shape[0]):
            for j in range(z.shape[2]):
                zbar[i,:,j] = self.h_np(middle_mu[i], lis[i,:,j])
        # calculate innovation and wrap as needed
        v = z[:,:,:,:2] - zbar
        # print(v[~np.isnan(v)].shape)
        temp = v[:,:,:,1]
        while np.any( v[:,:,:,1] > np.pi ):
            v[ v[:,:,:,1] > np.pi ] -= 2*np.pi
        while np.any( v[:,:,:,1] < -np.pi ):
            v[ v[:,:,:,1] < -np.pi ] += 2*np.pi


        # put together predict data
        temp = [torch.tensor(predict_mu_diff)] + [torch.tensor(data['start'][i]) for i in range(2,5)]
        self.px_train = torch.cat(temp, 2).float().cuda()
        temp = [torch.tensor(data['middle'][i]) for i in range(2,5)]
        self.py_train = torch.cat(temp, 2).float().cuda()

        # put together update data
        self.ux1_train = torch.tensor(v).float().cuda()
        temp = [torch.tensor(data['middle'][i]) for i in range(2,5)]
        self.ux2_train = torch.cat(temp, 2).float().cuda()
        temp = [torch.tensor(data['end'][i]-data['middle'][i]) for i in range(1,5)]
        self.uy_train = torch.cat(temp, 2).float().cuda()

        # put together train all data
        temp = [torch.tensor(data['start'][i]) for i in range(1,5)]
        self.allx_train = torch.cat(temp, 2).float().cuda()
        self.us = torch.tensor(data['us']).float().cuda()
        self.zs = torch.tensor(z[:,:,:,:2]).float().cuda()
        self.mlandmarks = torch.tensor(lis).float().cuda()
        temp = [torch.tensor(data['end'][i]) for i in range(1,5)]
        self.ally_train = torch.cat(temp, 2).float().cuda()

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

    def train_predict(self, idx):
        return self.px_train[idx], self.py_train[idx]

    def train_update(self, idx):
        return self.ux1_train[idx], self.ux2_train[idx], self.uy_train[idx]

    def train_all(self, idx):
        data = [self.allx_train[idx], self.us[idx], self.zs[idx], self.mlandmarks[idx], self.ally_train[idx]]
        return [torch.transpose(i, 0, 1) for i in data]

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

    def f_np(self, mu, u):
        x = mu[:,0] + u[:,1]*np.cos(mu[:,2] + u[:,0])
        y = mu[:,1] + u[:,1]*np.sin(mu[:,2] + u[:,0])
        theta = mu[:,2] + u[:,0] + u[:,2]
        return np.stack((x,y,theta), axis=1)

    def h_np(self, mu, l):
        dx = l[:,0]-mu[:,0]
        dy = l[:,1]-mu[:,1]
        r = np.sqrt( dx**2 + dy**2 )
        b = np.arctan2(dy, dx) - mu[:,2]
        return np.stack((r,b), 1)

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, input_data, output_data):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        layers = [nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU()]
        for i in range(n_layers):
            layers.append( nn.Linear(self.hidden_size, self.hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(self.hidden_size, self.output_size) )
        
        self.net = nn.Sequential(*layers)

        self.norm_in = Normalize(input_data)
        self.norm_out = Normalize(output_data)

    def forward(self, x, norm_out=True):
        #do some fancy stuff to make sure it's symmetric in the end
        x = self.net(self.norm_in(x))
        if norm_out:
            x = self.norm_out(x)
        return x

class ResNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, input_data, output_data):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.act = nn.ReLU()
        self.in_layer = nn.Linear(self.input_size, self.hidden_size)
        self.layers = []
        for i in range(n_layers//2):
            temp = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size), nn.Linear(self.hidden_size, self.hidden_size)])
            self.layers.append(temp)
        self.layers = nn.ModuleList(self.layers)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size)

        self.norm_in = Normalize(input_data)
        self.norm_out = Normalize(output_data)

        # def init_weights(m):
        #     if type(m) == nn.Linear:
        #         torch.nn.init.kaiming_uniform_(m.weight)
        #         m.bias.data.fill_(0.0)
        # self.apply(init_weights)

    def forward(self, x, norm_out=True):
        # normalize and do first layer
        x = self.norm_in(x)
        x = self.act( self.in_layer(x) )

        #pass through residual layers
        for layer in self.layers:
            temp = self.act( layer[0](x) )
            x = self.act( x + layer[1]( temp ))

        #output and normalize if asked for
        x = self.out_layer(x)
        if norm_out:
            x = self.norm_out(x)
        return x


if __name__ == "__main__":
    test = OdometryData("exact_data_4.pkl")