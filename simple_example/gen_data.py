from deepfilters.filters import ParticleFilter
from deepfilters.systems import OdometrySystem

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
start_mu = np.zeros((times*steps, 3))
start_cov = np.zeros((times*steps, 3, 3))
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
    all_x[steps*i:steps*(i+1)] = xs

    n = 1000
    # pf = ParticleFilter(sys, N=n, mean=np.array([180, 50, 0]), cov=np.diag([200, 200, np.pi/4]), pz_x=sys.pz_x)
    pf = ParticleFilter(sys, N=n, mean=xs[0], cov=np.diag([0.1, 0.1, 0.1]), pz_x=sys.pz_x)
    all_particles = []

    for j, (u, z) in enumerate(zip(us, zs)):
        # save where we started at each time step
        # somewhat redundant, but makes things a bit easier later
        start_mu[steps*i+j] = pf.particles.mean(axis=0)
        start_cov[steps*i+j] = np.cov(pf.particles, rowvar=False)

        # predict step and save
        p = pf.predict(u)
        p_mu[steps*i+j]  = p.mean(axis=0)
        p_cov[steps*i+j] = np.cov(p, rowvar=False)

        # clean measurements
        temp = z[ ~np.isnan(z).any(axis=1) ]
        ls = l[ temp[:,2].astype('int')-1 ]
        meas = temp[:,:2]

        # update step and save
        if meas.shape != (0,):
            try:
                p = pf.update(meas, ls)
            except:
                bad_time.append(i)
                bad_step.append(j)
        u_mu[steps*i+j]  = p.mean(axis=0)
        u_cov[steps*i+j] = np.cov(p, rowvar=False)

        bar.update(1)
    
    import matplotlib.pyplot as plt
    plt.plot(xs[:,0], xs[:,1], label="Actual")
    plt.plot(start_mu[steps*i:steps*(i+1),0], start_mu[steps*i:steps*(i+1),1], label="PF")
    plt.legend()
    plt.show()
    
np.savez("odometry_particle_shifted.npz", p_mu=p_mu, p_cov=p_cov, u_mu=u_mu, u_cov=u_cov, z=all_z, u=all_u, start_mu=start_mu, start_cov=start_cov)
print(bad_time)
print(bad_step)