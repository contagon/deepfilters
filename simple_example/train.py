import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from networks import *

def train_predict_sigma(data, model, lr, epochs, plot=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    objective = nn.L1Loss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split, leave=True, position=0)
    for epoch in range(epochs):
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            sig_train, diff_train, y_train = data.train_predict_sigma(idx)
            x_model = model(sig_train, diff_train)

            loss = objective(x_model.view(-1, 9), y_train.view(-1, 9))
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Train Loss: {loss.item()}")

            if batch % len(idxs)/10 == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for i, val_idx in enumerate(val_idxs):
                    sig_train, diff_train, y_train = data.train_predict_sigma(val_idx)
                    x_model = model(sig_train, diff_train)

                    loss = objective(x_model.view(-1, 9), y_train.view(-1, 9))
                    temp_val_loss[i] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        plt.semilogy(np.arange(len(train_loss)), train_loss)
        plt.semilogy(np.arange(len(val_loss)), val_loss)
        plt.title("Predict Sigma Loss")
        plt.show()

def train_update_mu(data, model, lr, epochs, plot=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    objective = nn.L1Loss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split, leave=True, position=0)
    for epoch in range(epochs):
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            sig_train, innov_train, y_train = data.train_update_mu(idx)
            x_model = torch.zeros_like(y_train)
            for i in range(innov_train.shape[1]):
                mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool()
                x_model[mask] += model(sig_train[mask], innov_train[mask][:,i,:])

            loss = objective(x_model, y_train)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Train Loss: {loss.item()}")

            if batch % len(idxs)/10 == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for j, val_idx in enumerate(val_idxs):
                    sig_train, innov_train, y_train = data.train_update_mu(val_idx)
                    x_model = torch.zeros_like(y_train)
                    for i in range(innov_train.shape[1]):
                        mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool()
                        x_model[mask] += model(sig_train[mask], innov_train[mask][:,i,:])

                    loss = objective(x_model, y_train)
                    temp_val_loss[j] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        plt.semilogy(np.arange(len(train_loss)), train_loss)
        plt.semilogy(np.arange(len(val_loss)), val_loss)
        plt.title("Update Mu Loss")
        plt.show()

def train_update_sigma(data, model, lr, epochs, plot=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    objective = nn.L1Loss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split, leave=True, position=0)
    for epoch in range(epochs):
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            sig_train, innov_train, y_train = data.train_update_sigma(idx)
            # for i in range(innov_train.shape[1]):
            i = 0
            mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool()
            x_model = torch.zeros_like(sig_train)
            x_model[mask] = model(sig_train[mask], innov_train[mask][:,i,:])

            loss = objective(x_model.view(-1, 9), y_train.view(-1, 9))
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Train Loss: {loss.item()}")

            # if batch % len(idxs)/10 == 0:
            #     train_loss.append(loss.item())

            #     val_idxs = data.rand_test_idx
            #     temp_val_loss = np.zeros(len(val_idxs))
            #     for i, val_idx in enumerate(val_idxs):
            #         sig_train, innov_train, y_train = data.train_update_sigma(val_idx)
            #         for i in range(innov_train.shape[1]):
            #             mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool()
            #             sig_train[mask] = model(sig_train[mask], innov_train[mask][:,i,:])

            #         loss = objective(sig_train.view(-1, 9), y_train.view(-1, 9))
            #         temp_val_loss[i] = loss.item()
            #     val_loss.append( temp_val_loss.mean() )

    if plot:
        plt.semilogy(np.arange(len(train_loss)), train_loss)
        plt.semilogy(np.arange(len(val_loss)), val_loss)
        plt.title("Predict Sigma Loss")
        plt.show()

def main():
    # we'll train each network individually for a while first
    # Then we'll work on training as a sequence to help it stand on it's own
    data = OdometryData("odometry_particle_data.npz", split=8000)

    #first PredictSigma
    # p_sigma = Sigma(3, 3, 64, 24).cuda()
    # train_predict_sigma(data, p_sigma, 1e-3, 5)

    #next UpdateMu
    # u_mu = UpdateMu(3, 2, 3, 64, 24).cuda()
    # train_update_mu(data, u_mu, 1e-4, 5)

    #finally UpdateSigma
    u_sigma = Sigma(3, 2, 64, 24).cuda()
    train_update_sigma(data, u_sigma, 1e-4, 1)

    # cov = data.p_cov.cpu().numpy()
    # # 31
    # cov_mean = cov.reshape(-1, 200, 3, 3).mean(axis=1)
    # print(cov.mean(axis=0))
    # for i in range(3):
    #     plt.hist(data.p_cov[31*200:32*200,i,i].cpu())
    #     plt.show()
    
if __name__ == "__main__":
    main()