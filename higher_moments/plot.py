import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from networks import *
import sys
from deepfilters.filters import ExtendKalmanFilter, ParticleFilter
import seaborn as sns

def remove_ticks():
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False,
    left=False,
    right=False) # labels along the bottom edge are off

# load data
idx = int( sys.argv[2] ) #9009
data = OdometryData("exact_data_4.pkl", split=8000)
mat_data = loadmat("data.mat")['data'][0,idx]
# data = OdometryData("unknown_data_4.pkl", split=8000)

#set up system 
xs = mat_data['realRobot'][0,0].T
us = mat_data['noisefreeControl'][0,0].T
zs = mat_data['realObservation'][0,0].T
landmarks = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21],
                [0, 0, 0, 0, 292, 292, 292, 292]]).T

# setup covariances
alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
beta = np.array([10, np.deg2rad(10)])**2

#setup system
system = OdometrySystem(alphas, beta)

#######################################################################################
####################                  Ground Truth                 ####################
#######################################################################################
mu_actual = mat_data['realRobot'][0,0].T

#######################################################################################
####################             Extended Kalman Filter            ####################
#######################################################################################
ekf = ExtendKalmanFilter(system, xs[0], np.eye(3)*.01)
for u,z in zip(us, zs):
    ekf.predict(u)

    temp = z[ ~np.isnan(z).any(axis=1) ]
    ls = landmarks[ temp[:,2].astype('int')-1 ]
    meas = temp[:,:2]

    for zi, li in zip(meas, ls):
        ekf.update(zi, li)

mu_ekf = np.array(ekf.mus[1:])
temp = np.triu_indices(3)
sig_ekf = np.array(ekf.sigmas[1:])[:,temp[0], temp[1]]

#######################################################################################
####################                 Particle Filter               ####################
#######################################################################################
n = 1000
pf = ParticleFilter(system, N=n, mean=xs[0], cov=np.eye(3)*.01, pz_x=system.pz_x)
all_particles = []
for u, z in zip(us, zs):
    pf.predict(u)

    temp = z[ ~np.isnan(z).any(axis=1) ]
    ls = landmarks[ temp[:,2].astype('int')-1 ]
    meas = temp[:,:2]

    if meas.shape != (0,):
        all_particles.append( pf.update(meas, ls) )
        
all_particles = np.array(all_particles)


#######################################################################################
####################                Neural Networks                ####################
#######################################################################################

# load models in
all = slice(None,None)
n_m = 3+6+10+15
pnn = ResNetwork(n_m, n_m-3, 64, 12, *data.train_predict(all)).cuda().eval()
x1, x2, y = data.train_update(all)
x = torch.cat([(x1[:,:,0,:]+x1[:,:,1,:])/2, x2], 2)
unn = ResNetwork(n_m-3+2, n_m, 64, 12, x, y).cuda().eval()

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
    if (torch.abs(v) > 10000).any():
        break
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
mu_pf = y_train[:,0,:].cpu().detach().numpy()
mu_nn = np.array(m_result).squeeze()

#######################################################################################
####################                Plot Everything                ####################
#######################################################################################

sns.set_style("whitegrid")
fig = plt.figure(figsize=(8.5,3.5))
gridspec.GridSpec(2, 7)
lines = []
ttl = 100

# plot particle training data
alpha = .01
plt.subplot2grid((2,7), (0,0), 2, 2)
plt.plot(mu_actual[:100,0], mu_actual[:ttl,1], label="Actual Trajectory", c='k')
for i in range(n):
    if i == 1:
        handle = plt.scatter(all_particles[:ttl,i,0], all_particles[:ttl,i,1], alpha=alpha, s=1, c='r', label="Particles")
    else:
        plt.scatter(all_particles[:ttl,i,0], all_particles[:ttl,i,1], alpha=alpha, s=1, c='r')
plt.scatter(landmarks[:,0], landmarks[:,1], s=20)
plt.title("(a)")
remove_ticks()
handle.set_alpha(1)

# plot resulting trajectory
plt.subplot2grid((2,7), (0,2), 2, 2)
plt.plot(mu_actual[:ttl,0], mu_actual[:ttl,1], label="Actual Trajectory", c='k')
plt.plot(mu_pf[:ttl,0], mu_pf[:ttl,1], label="PF")
plt.plot(mu_nn[:ttl,0], mu_nn[:ttl,1], label="ANN")
plt.plot(mu_ekf[:ttl,0], mu_ekf[:ttl,1], label="EKF")
plt.scatter(landmarks[:,0], landmarks[:,1], s=20)
remove_ticks()
plt.title("(b)")
#setup legend
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handle] + handles
labels = ["Particles"] + labels
plt.legend(handles, labels, ncol=5, bbox_to_anchor=(0.8,0.1), bbox_transform=plt.gcf().transFigure)

# Plot Sigmas
t = np.arange(ttl)
for i in range(2):
    for j in range(3):
        plt.subplot2grid((2,7), (i,4+j), 1, 1)
        plt.plot(t, mu_pf[:ttl,i*3+j+3], label="PF Sigma")
        plt.plot(t, mu_nn[:ttl,i*3+j+3], label="Network Result")
        plt.plot(t, sig_ekf[:ttl,i*3+j], label="EKF Result")
        remove_ticks()
        if i == 0 and j == 1:
            plt.title(("(c)"))
# plt.legend()
plt.subplots_adjust(top=0.935,
bottom=0.11,
left=0.01,
right=0.99,
hspace=0.05,
wspace=0.05)
plt.show()