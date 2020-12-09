import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networks import *
import sys

# load data
n = 9400
data = OdometryData("odometry_particle_shifted.npz", split=8000)
train_mu, train_sig, us, zs, landmarks, y_mu, y_sig = data.train_all(n)
m = train_mu[0].unsqueeze(0)
s = train_sig[0].unsqueeze(0)
m_result = [m.cpu().detach().numpy()]
s_result = [s.cpu().detach().numpy()]
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

# iterate through
for u, zi, li in zip(us, zs, landmarks):
    # predict mu
    mi = m.clone()
    m = data.f(mi, u.unsqueeze(0))

    #predict sigma
    s = p_sigma(s, m-mi, norm_out=False)

    # calculate innovation
    vs = []
    for z, l in zip(zi, li):
        if torch.isnan(l).any():
            continue
        zbar = data.h(m, l.unsqueeze(0))
        v = z - zbar
        while v[:,1] > np.pi:
            v[:,1] -= 2*np.pi
        while v[:,1] < -np.pi:
            v[:,1] += 2*np.pi
        vs.append(v)
    if vs != []:
        vs = torch.cat(vs, 0)

    #update mu
    for v in vs:
        m += u_mu(s, v.unsqueeze(0), norm_out=False)
 
    
    # update sigma
    for v in vs:
        s = u_sigma(s, v.unsqueeze(0), norm_out=False)

    # calculate loss
    m_result.append( m.cpu().detach().numpy() )
    s_result.append( s.cpu().detach().numpy() )

# clean up results
mu_actual = loadmat("data.mat")['data'][0,n]['realRobot'][0,0].T
mu_pf = y_mu.cpu().detach().numpy()
mu_network = np.array(m_result).squeeze()

s_pf = y_sig.cpu().detach().numpy()
s_network = np.array(s_result).squeeze()

# # plot trajectory
# plt.plot(mu_actual[:,0], mu_actual[:,1], label="Ground Truth")
# plt.plot(mu_pf[:,0], mu_pf[:,1], label="PF Result")
# plt.plot(mu_network[:,0], mu_network[:,1], label="Network Result")
# plt.legend()
# plt.show()

# #plot theta
# t = np.arange(200)
# plt.plot(t, mu_actual[:,2], label="Actual")
# plt.plot(np.arange(199), mu_pf[:,2], label="PF Result")
# plt.plot(t, mu_network[:,2], label="Network Result")
# plt.legend()
# plt.show()

t = np.arange(200)
fig, axs = plt.subplots(2,3)
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.plot(np.arange(199), s_pf[:,i], label="PF")
    ax.plot(t, s_network[:,i], label="PF Result")
plt.legend()
plt.show()