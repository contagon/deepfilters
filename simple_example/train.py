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

            loss = objective(x_model, y_train)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"PS, Train Loss: {loss.item()}")

            if batch % (len(idxs)/2) == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for i, val_idx in enumerate(val_idxs):
                    sig_train, diff_train, y_train = data.train_predict_sigma(val_idx)
                    x_model = model(sig_train, diff_train)

                    loss = objective(x_model, y_train)
                    temp_val_loss[i] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.semilogy(np.arange(len(val_loss)), val_loss)
        ax.set_title("Predict Sigma Loss")
        plt.draw()
        plt.pause(0.001)

def train_update_mu(data, model, lr, epochs, plot=True):
    opt = optim.Adam(model.parameters(), lr=lr)
    objective = nn.MSELoss().cuda()

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
                mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                x_model[mask] += model(sig_train[mask], innov_train[mask][:,i,:])

            #scale so outputs are same size ish
            x_model[:,2]*= 10
            y_train_temp = y_train.clone()
            y_train_temp[:,2] *= 10

            loss = objective(x_model, y_train_temp)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"UM, Train Loss: {loss.item()}")

            if batch % (len(idxs)/2) == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for j, val_idx in enumerate(val_idxs):
                    sig_train, innov_train, y_train = data.train_update_mu(val_idx)
                    x_model = torch.zeros_like(y_train)
                    for i in range(innov_train.shape[1]):
                        mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                        x_model[mask] += model(sig_train[mask], innov_train[mask][:,i,:])

                    x_model[:,2]*= 10
                    y_train_temp = y_train.clone()
                    y_train_temp[:,2] *= 10

                    loss = objective(x_model, y_train)
                    temp_val_loss[j] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.semilogy(np.arange(len(val_loss)), val_loss)
        ax.set_title("Update Mu Loss")
        plt.draw()
        plt.pause(0.001)

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
            for i in range(innov_train.shape[1]):
                mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                sig_train[mask] = model(sig_train[mask], innov_train[mask][:,i,:])

            loss = objective(sig_train, y_train)
            loss.backward()
            opt.step()

            t.update(1)
            t.set_description(f"US, Train Loss: {loss.item()}")
            if batch % (len(idxs)//2) == 0:
                train_loss.append(loss.item())

                val_idxs = data.rand_test_idx
                temp_val_loss = np.zeros(len(val_idxs))
                for j, val_idx in enumerate(val_idxs):
                    sig_train, innov_train, y_train = data.train_update_sigma(val_idx)
                    for i in range(innov_train.shape[1]):
                        mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
                        sig_train[mask] = model(sig_train[mask], innov_train[mask][:,i,:])
                        
                    loss = objective(sig_train, y_train)
                    temp_val_loss[j] = loss.item()
                val_loss.append( temp_val_loss.mean() )

    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.semilogy(np.arange(len(val_loss)), val_loss)
        ax.set_title("Update Sigma Loss")
        plt.draw()
        plt.pause(0.001)

def train_all(data, model_psig, model_umu, model_usig, lr, epochs, batch_size=100, teacher_forcing=0, plot=True):
    objective = nn.L1Loss(reduction='none').cuda()
    # params = list(model_psig.parameters()) + list(model_umu.parameters()) + list(model_usig.parameters())
    # opt = optim.Adam(params, lr=lr)
    params = list(model_psig.parameters()) + list(model_usig.parameters())
    opt_sig = optim.Adam(params, lr=lr)
    opt_mu = optim.Adam(model_umu.parameters(), lr=lr)

    val_loss = []
    train_loss = []
    t = tqdm(total=epochs*data.split/batch_size, leave=True, position=0)
    #iterate through epochs
    for epoch in range(epochs):
        idxss = data.rand_train_idx.reshape((-1, batch_size))
        #iterate through each batch of epochs
        for batch, idxs in enumerate(idxss):
            opt_mu.zero_grad()
            opt_sig.zero_grad()

            # decide if we should do teacher forcing
            if np.random.random() < teacher_forcing:
                force = True
            else:
                force = False

            #run deepfilter on each timestep, adding together losses (if they're not too huge..)
            temp = []
            for idx in idxs:
                # train_mu, train_sig, us, zs, landmarks, y_mu, y_sig = data.train_all(idx)
                temp.append( data.train_all(idx) )
            temp = list(zip(*temp))
            train_mu, train_sig, us, zs, landmarks, y_mu, y_sig = tuple([torch.stack(i, 1) for i in temp])

            loss_sig = 0
            loss_mu = 0
            m = train_mu[0]
            s = train_sig[0]
            for real_m, real_s, u, zi, li, ym, ys in zip(train_mu, train_sig, us, zs, landmarks, y_mu, y_sig):
                #if we're teacher forcing this batch
                if force:
                    m = real_m.clone()
                    s = real_s.clone()
                else:
                    m = m.detach()
                    s = s.detach()
                    
                # predict mu
                mi = m.clone()
                m = data.f(mi, u)
                #predict sigma
                s = model_psig(s, m-mi)

                # calculate innovation
                vs = []
                for z, l in zip(zi.transpose(0,1), li.transpose(0,1)):
                    zbar = data.h(m, l)
                    v = z - zbar
                    while (v[:,1] > np.pi).any():
                        v[ v[:,1] > np.pi ] -= 2*np.pi
                    while (v[:,1] < -np.pi).any():
                        v[ v[:,1] < -np.pi ] += 2*np.pi
                    vs.append(v)

                if vs != []:
                    vs = torch.stack(vs, 1)

                #update mu
                for v in vs.transpose(0,1):
                    mask = ~torch.isnan(v).byte().any(axis=1).bool().detach()
                    m[mask] += model_umu(s[mask], v[mask])

                # update sigma
                for v in vs.transpose(0,1):
                    mask = ~torch.isnan(v).byte().any(axis=1).bool().detach()
                    s[mask] = model_usig(s[mask], v[mask])

                loss_mu  += objective(ym, m).mean(axis=1)
                loss_sig += objective(ys, s).mean(axis=1)

            loss = loss_mu.mean()/200 + loss_sig.mean()/200
            # if loss_mu.mean() > 5000:
                # print(loss_mu)
            loss.backward()
            # opt_mu.step()
            opt_sig.step()
            t.update(1)
            train_loss.append(loss.item())
            t.set_description(f"Mu Loss: {loss_mu.mean().item()/200}, Sigma Loss: {loss_sig.mean().item()/200}")



    if plot:
        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(train_loss)), train_loss)
        ax.set_title("Total Together Loss")
        plt.draw()
        plt.pause(0.001)

def main(filename):
    # we'll train each network individually for a while first
    # Then we'll work on training as a sequence to help it stand on it's own
    data = OdometryData("odometry_particle_shifted.npz", split=8000)

    # #first PredictSigma
    p_sigma = Sigma(3, 3, 64, 12).cuda()
    train_predict_sigma(data, p_sigma, 1e-3, 5, plot=True)

    # # #next UpdateMu
    u_mu = UpdateMu(3, 2, 3, 64, 12).cuda()
    train_update_mu(data, u_mu, 1e-3, 3, plot=True)

    # # #finally UpdateSigma
    u_sigma = Sigma(3, 2, 64, 12).cuda()
    train_update_sigma(data, u_sigma, 1e-3, 5, plot=True)

    torch.save({"p_sigma": p_sigma.state_dict(),
                "u_mu": u_mu.state_dict(),
                "u_sigma": u_sigma.state_dict()}, filename+"temp")

    # # load models in
    # p_sigma = Sigma(3, 3, 64, 12).cuda()
    # u_mu = UpdateMu(3, 2, 3, 64, 12).cuda()
    # u_sigma = Sigma(3, 2, 64, 12).cuda()

    # # restore weights
    # models = torch.load(filename+"temp")
    # p_sigma.load_state_dict(models['p_sigma'])
    # u_mu.load_state_dict(models['u_mu'])
    # u_sigma.load_state_dict(models['u_sigma'])

    #train them all together!
    # train_all(data, p_sigma, u_mu, u_sigma, 1e-3, 2, teacher_forcing=1, batch_size=200)
    train_all(data, p_sigma, u_mu, u_sigma, 1e-3, 5, teacher_forcing=0.5, batch_size=200)
    train_all(data, p_sigma, u_mu, u_sigma, 1e-3, 5, teacher_forcing=0, batch_size=200)

    torch.save({"p_sigma": p_sigma.state_dict(),
                "u_mu": u_mu.state_dict(),
                "u_sigma": u_sigma.state_dict()}, filename)

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1])