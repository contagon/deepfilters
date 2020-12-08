import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networks import *
import sys

# load models in
p_sigma = Sigma(3, 3, 64, 12).cuda().eval()
u_mu = UpdateMu(3, 2, 3, 64, 12).cuda().eval()
u_sigma = Sigma(3, 2, 64, 12).cuda().eval()

# restore weights
models = torch.load(sys.argv[1])
p_sigma.load_state_dict(models['p_sigma'])
u_mu.load_state_dict(models['u_mu'])
u_sigma.load_state_dict(models['u_sigma'])

# load data
n = 9000
data = OdometryData("odometry_particle_shifted.npz", split=8000)
train_mu, train_sig, us, zs, landmarks, y_mu, y_sig = data.train_all(n)
m = train_mu[0].unsqueeze(0)
s = train_sig[0].unsqueeze(0)
m_result = [m.cpu().detach().numpy()]

# iterate through
for u, zi, li in zip(us, zs, landmarks):
    # predict mu
    mi = m.clone()
    m = data.f(mi, u.unsqueeze(0))

    #predict sigma
    s = p_sigma(s, m-mi)

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
        m += u_mu(s, v.unsqueeze(0))
        # print(v)
        # print(u_mu(s, v.unsqueeze(0)))

    # update sigma
    for v in vs:
        s = u_sigma(s, v.unsqueeze(0))

    # calculate loss
    m_result.append( m.cpu().detach().numpy() )

# clean up results
mu_actual = loadmat("data.mat")['data'][0,n]['realRobot'][0,0].T
mu_pf = y_mu.cpu().detach().numpy()
mu_network = np.array(m_result).squeeze()

# plot things!
plt.plot(mu_actual[:,0], mu_actual[:,1], label="Ground Truth")
plt.plot(mu_pf[:,0], mu_pf[:,1], label="PF Result")
plt.plot(mu_network[:,0], mu_network[:,1], label="Network Result")
plt.legend()
plt.show()