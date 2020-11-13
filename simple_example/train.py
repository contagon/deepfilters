import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from networks import *

def train_predict_sigma(data, model, lr, epochs, plt=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    objective = nn.MSELoss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*10000, leave=True, position=0)
    for epoch in range(epochs):
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            x_train, y_train = data.train_predict_sigma(idx)
            x_model = model(x_train)

            loss = objective(x_model.view(-1, 9), y_train.view(-1, 9))
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Train Loss: {loss.item()}")

            if batch % 1000 == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for i, val_idx in enumerate(val_idxs):
                    x_train, y_train = data.train_predict_sigma(idx)
                    x_model = model(x_train)

                    loss = objective(x_model.view(-1, 9), y_train.view(-1, 9))
                    temp_val_loss[i] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plt:
        plt.semilogy(np.arange(len(train_loss)), train_loss)
        plt.semilogy(np.arange(len(val_loss)), val_loss)
        plt.show()


def main():
    # we'll train each network individually for a while first
    # Then we'll work on training as a sequence to help it stand on it's own
    data = OdometryData("odometry_particle_data.npz")

    #first PredictSigma
    u_sigma = Sigma(3, 50, 5).cuda()
    train_predict_sigma(data, u_sigma, 1e-4, 1)

    #next UpdateMu
    u_sigma = UpdateMu(3, 50, 5).cuda()
    train_predict_sigma(data, u_sigma, 1e-4, 1)

    #finally UpdateSigma
    u_sigma = Sigma(3, 50, 5).cuda()
    train_predict_sigma(data, u_sigma, 1e-4, 1)

    
    
if __name__ == "__main__":
    main()