import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from networks import *
import sys
torch.set_printoptions(sci_mode=False)

def train_predict(data, model, lr, epochs, plot=True):
    norm = torch.ones(31).cuda()
    norm[0] = 1.5
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    objective = nn.L1Loss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split, leave=True, position=0)
    # iterate through epochs
    for epoch in range(epochs):
        # iterate through all batches (each is a trajectory of 200)
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            x_train, y_train = data.train_predict(idx)
            y_model = model(x_train)

            y_train = model.norm_out(y_train)
            loss = objective(y_model*norm, y_train*norm)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Predict, Train Loss: {loss.item()}")

            # run validation
            if batch % (len(idxs)/2) == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for i, val_idx in enumerate(val_idxs):
                    x_train, y_train = data.train_predict(idx)
                    y_model = model(x_train)

                    y_train = model.norm_out(y_train)
                    loss = objective(y_model, y_train)
                    temp_val_loss[i] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.semilogy(np.arange(len(val_loss)), val_loss)
        ax.set_title("Predict Loss")
        plt.draw()
        plt.pause(0.001)

def train_update(data, model, lr, epochs, plot=True):
    norm = torch.ones(34).cuda()
    # norm[0:3] = 1.5
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    objective = nn.L1Loss().cuda()

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split, leave=True, position=0)
    # iterate through epochs
    for epoch in range(epochs):
        # iterate through all batches (each is a trajectory of 200)
        idxs = data.rand_train_idx
        for batch, idx in enumerate(idxs):
            opt.zero_grad()

            # push through model
            innov_train, moments_train, y_train = data.train_update(idx)
            y_model = torch.zeros_like(y_train)
            for i in range(innov_train.shape[1]):
                mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                x_train = torch.cat([innov_train[:,i,:], moments_train], 1)
                y_model[mask] += model(x_train[mask])

            # calculate loss and step
            y_train = model.norm_out(y_train)
            loss = objective(y_model*norm, y_train*norm)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"Update, Train Loss: {loss.item()}")

            # run validation
            if batch % (len(idxs)/2) == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                # run through all validation batches
                for j, val_idx in enumerate(val_idxs):
                    innov_train, moments_train, y_train = data.train_update(idx)
                    y_model = torch.zeros_like(y_train)
                    for i in range(innov_train.shape[1]):
                        mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                        x_train = torch.cat([innov_train[:,i,:], moments_train], 1)
                        y_model[mask] += model(x_train[mask])

                    y_train = model.norm_out(y_train)
                    loss = objective(y_model, y_train)
                    temp_val_loss[j] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.semilogy(np.arange(len(val_loss)), val_loss)
        ax.set_title("Update Loss")
        plt.draw()
        plt.pause(0.001)

def train_all(data, pnn, unn, lr, epochs, plot=True, batch_size=100, teacher_forcing=0):
    outputs = data.train_all(slice(None))[-1]
    OutNorm = Normalize(outputs)

    norm = torch.ones(34).cuda()
    norm[:3] = 2
    objective = nn.L1Loss(reduction='none').cuda()   
    params = list(pnn.parameters()) + list(unn.parameters())
    opt = optim.Adam(params, lr=lr, weight_decay=0)

    train_loss = []
    t = tqdm(total=epochs*data.split/batch_size, leave=True, position=0)
    #iterate through epochs
    for epoch in range(epochs):
        idxss = data.rand_train_idx.reshape((-1, batch_size))
        #iterate through each batch of epochs
        for batch, idxs in enumerate(idxss):
            opt.zero_grad()

            # decide if we should do teacher forcing
            if np.random.random() < teacher_forcing:
                force = True
            else:
                force = False

            x_train, us, zs, ls, y_train = data.train_all(idxs)
            moment = x_train[0]
            loss = 0
            for real_moment, u, z, l, y in zip(x_train, us, zs, ls, y_train):
                #if we're teacher forcing or not
                if force:
                    moment = real_moment.clone()
                else:
                    moment = moment.detach()

                #predict mu
                mu = data.f(moment[:,:3], u)
                diff_mu = mu - moment[:,:3]
                
                #predict rest of moments
                temp = pnn( torch.cat([diff_mu, moment[:,3:]], 1), norm_out=False)
                moment = torch.cat([mu, temp], 1)

                #calculate innovation
                vs = []
                for zi, li in zip(z.transpose(0,1), l.transpose(0,1)):
                    zbar = data.h(moment[:,:3], li)
                    v = zi - zbar
                    while (v[:,1] > np.pi).any():
                        v[ v[:,1] > np.pi ] -= 2*np.pi
                    while (v[:,1] < -np.pi).any():
                        v[ v[:,1] < -np.pi ] += 2*np.pi
                    vs.append(v)

                if vs != []:
                    vs = torch.stack(vs, 1)

                # update step
                temp = torch.zeros_like(y)
                for v in vs.transpose(0,1):
                    mask = ~torch.isnan(v).byte().any(axis=1).bool().detach()
                    temp[mask] += unn( torch.cat([v, moment[:,3:]], 1)[mask], norm_out=False)
                moment += temp

                if (torch.abs(temp) > 1000).any():
                    print( (torch.abs(temp) > 1000).nonzero() )
                # add to loss
                loss += objective(OutNorm(moment), OutNorm(y)).mean(axis=1)

            #step optimizer
            loss = loss.mean() / 200
            loss.backward()
            opt.step()

            t.update(1)
            train_loss.append(loss.item())
            t.set_description(f"Train All, Loss: {loss.item()}")

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.set_title("Total Together Loss")
        plt.draw()
        plt.pause(0.001)


def main(filename):
    # we'll train each network individually for a while first
    # Then we'll work on training as a sequence to help it stand on it's own
    # data = OdometryData("unknown_data_4.pkl", split=9000)
    data = OdometryData("exact_data_4.pkl", split=9000)
    all = slice(None)

    # make models
    n_m = 3+6+10+15
    pnn = ResNetwork(n_m, n_m-3, 64, 12, *data.train_predict(all)).cuda()
    x1, x2, y = data.train_update(all)
    x = torch.cat([(x1[:,:,0,:]+x1[:,:,1,:])/2, x2], 2)
    unn = ResNetwork(n_m-3+2, n_m, 64, 12, x, y).cuda()

    load = False
    if not load:
        # train models
        train_predict(data, pnn, 1e-3, 2)
        train_update(data, unn, 1e-3, 2)
        torch.save({"pnn": pnn.state_dict(),
                    "unn": unn.state_dict()}, filename+"temp")
    else:
        # load models
        models = torch.load(sys.argv[1]+"temp")
        pnn.load_state_dict(models['pnn'])
        unn.load_state_dict(models['unn'])

    # save networks
    train_all(data, pnn, unn, 1e-3, 2, batch_size=200, teacher_forcing=1)
    train_all(data, pnn, unn, 1e-3, 1, batch_size=200, teacher_forcing=0.5)
    train_all(data, pnn, unn, 1e-3, 2, batch_size=200, teacher_forcing=0)
    
    torch.save({"pnn": pnn.state_dict(),
                    "unn": unn.state_dict()}, filename)
    

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1])