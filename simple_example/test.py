import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networks import *
import sys

# load data
n = 9400
data = OdometryData("odometry_particle_shifted.npz", split=8000)

# load models in
all = slice(None,None)
p_sigma = Sigma(3, 3, 64, 12, *data.train_predict_sigma(all)).cuda().eval()
u_mu = UpdateMu(3, 2, 3, 64, 12, *data.train_update_mu(all)).cuda().eval()
u_sigma = Sigma(3, 2, 64, 12, *data.train_update_sigma(all)).cuda().eval()
# restore weights
models = torch.load(sys.argv[1])
p_sigma.load_state_dict(models['p_sigma'])
u_mu.load_state_dict(models['u_mu'])
u_sigma.load_state_dict(models['u_sigma'])

sig_train, innov_train, y_train = data.train_update_sigma(n)
y_model = sig_train.clone()
for i in range(innov_train.shape[1]):
    mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
    sig_train[mask] = u_sigma(sig_train[mask], innov_train[:,i,:][mask], norm_out=False)
y_model = sig_train.clone()

i = 45
print("OG", sig_train[i])
print("Expected", y_train[i])
print("Got", y_model[i])

s_pf = y_train.cpu().detach().numpy()
s_nn = y_model.cpu().detach().numpy()

t = np.arange(200)
fig, axs = plt.subplots(2,3)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.plot(t, s_pf[:,i], label="PF")
    ax.plot(t, s_nn[:,i], label="NN")
plt.legend()
plt.show()