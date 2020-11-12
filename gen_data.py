from filters import ParticleFilter
from odometry import OdometrySystem

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time

# setup all data
data = loadmat("data.mat")['data']
steps = 200
times = 10000

# setup system
alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
beta = np.array([10, np.deg2rad(10)])**2
l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21],
                [0, 0, 0, 0, 292, 292, 292, 292]]).T
sys = OdometrySystem(alphas, beta)

# setup places to put things
p_mu = np.zeros((times*steps, 3))
p_cov = np.zeros((times*steps, 3, 3))
u_mu = np.zeros((times*steps,3))
u_cov = np.zeros((times*steps, 3, 3))
all_z = np.zeros((times*steps, 2, 3))
all_u = np.zeros((times*steps, 3))
all_x = np.zeros((times*steps, 3))

bar = tqdm(total=times*steps)

bad_time = []
bad_step = []

for i in range(times):
    xs = data[0,i]['realRobot'][0,0].T
    us = data[0,i]['noisefreeControl'][0,0].T
    zs = data[0,i]['realObservation'][0,0].T

    all_z[steps*i:steps*(i+1)] = zs
    all_u[steps*i:steps*(i+1)] = us
    all_u[steps*i:steps*(i+1)] = xs

    n = 1000
    pf = ParticleFilter(sys, N=n, mean=xs[0], cov=np.zeros((3,3)), pz_x=sys.pz_x)
    all_particles = []

    for j, (u, z) in enumerate(zip(us, zs)):
        # start = time()
        p = pf.predict(u)
        # print("PREDICT:", time()- start)
        p_mu[steps*i+j]  = p.mean(axis=0)
        p_cov[steps*i+j] = np.cov(p, rowvar=False)

        temp = z[ ~np.isnan(z).any(axis=1) ]
        ls = l[ temp[:,2].astype('int')-1 ]
        meas = temp[:,:2]

        if meas.shape != (0,):
            # start = time()
            try:
                p = pf.update(meas, ls)
            except:
                bad_time.append(i)
                bad_step.append(j)
            # print("UPDATE", time()-start)
        u_mu[steps*i+j]  = p.mean(axis=0)
        u_cov[steps*i+j] = np.cov(p, rowvar=False)

        bar.update(1)
    
np.savez("odometry_particle_data.npz", p_mu=p_mu, p_cov=p_cov, u_mu=u_mu, u_cov=u_cov, z=all_z, u=all_u)
print(bad_time)
print(bad_step)