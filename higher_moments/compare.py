import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networks import *
import sys

# load data
idx = 9400
data = OdometryData("exact_data_4.pkl", split=8000)

# load models in
all = slice(None,None)
n_m = 3+6+10+15
pnn = Network(n_m, n_m-3, 64, 12, *data.train_predict(all)).cuda().eval()
x1, x2, y = data.train_update(all)
x = torch.cat([(x1[:,:,0,:]+x1[:,:,1,:])/2, x2], 2)
unn = Network(n_m-3+2, n_m, 64, 12, x, y).cuda().eval()

#restore weights
models = torch.load(sys.argv[1])
pnn.load_state_dict(models['pnn'])
unn.load_state_dict(models['unn'])

x_train, us, zs, ls, y_train = data.train_all(slice(idx,idx+1))
moment = x_train[0]
m_result = []
for _, u, z, l, y in zip(x_train, us, zs, ls, y_train):
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

    #save for later
    m_result.append( moment.cpu().detach().numpy() )

# clean up results
mu_actual = loadmat("data.mat")['data'][0,idx]['realRobot'][0,0].T
m_pf = y_train[:,0,:].cpu().detach().numpy()
m_nn = np.array(m_result).squeeze()

# s_pf = y_sig.cpu().detach().numpy()
# s_network = np.array(s_result).squeeze()

# # plot trajectory
plt.plot(mu_actual[:,0], mu_actual[:,1], label="Ground Truth")
plt.plot(m_pf[:,0], m_pf[:,1], label="PF Result")
plt.plot(m_nn[:,0], m_nn[:,1], label="Network Result")
plt.legend()
plt.show()

# #plot theta
# t = np.arange(200)
# plt.plot(t, mu_actual[:,2], label="Actual")
# plt.plot(np.arange(199), m_pf[:,2], label="PF Result")
# plt.plot(t, m_nn[:,2], label="Network Result")
# plt.legend()
# plt.show()

t = np.arange(200)
fig, axs = plt.subplots(2,3)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.plot(t, m_pf[:,i+3], label="PF")
    ax.plot(t, m_nn[:,i+3], label="PF Result")
plt.legend()
plt.show()