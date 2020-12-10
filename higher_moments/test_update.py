import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networks import *
import sys

# load data
n = 9400
data = OdometryData("exact_data_4.pkl", split=8000)
all = slice(None,None)

# make models
n_m = 3+6+10+15
pnn = Network(n_m, n_m-3, 64, 12, *data.train_predict(all)).cuda()
x1, x2, y = data.train_update(all)
x = torch.cat([(x1[:,:,0,:]+x1[:,:,1,:])/2, x2], 2)
unn = Network(n_m-3+2, n_m, 64, 12, x, y).cuda()

# restore weights
models = torch.load(sys.argv[1])
pnn.load_state_dict(models['pnn'])
unn.load_state_dict(models['unn'])

# push through model
innov_train, moments_train, y_train = data.train_update(n)
y_model = torch.zeros_like(y_train)
for i in range(innov_train.shape[1]):
    mask = ~torch.isnan(innov_train[:,i,:]).byte().any(axis=1).bool().detach()
    x_train = torch.cat([innov_train[:,i,:], moments_train], 1)
    y_model[mask] += unn(x_train[mask], norm_out=False)

i = 45
print("OG", x_train[i])
print("Expected", y_train[i])
print("Got", y_model[i])

s_pf = y_train.cpu().detach().numpy()
s_nn = y_model.cpu().detach().numpy()

t = np.arange(200)
fig, axs = plt.subplots(3,3)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.plot(t, s_pf[:,i], label="PF")
    ax.plot(t, s_nn[:,i], label="NN")
plt.legend()
plt.show()