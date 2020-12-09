from deepfilters.filters import ParticleFilter
from deepfilters.systems import OdometrySystem

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations_with_replacement as cwr
import pickle

def calc_moment(data, i):
    first = data.mean(axis=0)
    N, n = data.shape
    if i == 1:
        return first
    
    #we'll calculate centered moments
    data = data - first

    #get all the combinations
    combos = list(cwr(np.arange(n), i))
    result = np.zeros(len(combos))
    for i, c in enumerate(combos):
        result[i] = np.product( data[:, list(c)], axis=1).sum() / N

    return result

# setup all data
data = loadmat("data.mat")['data']
steps = 200
times = 10000
moments = 4

# setup system
alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
beta = np.array([10, np.deg2rad(10)])**2
l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21],
                [0, 0, 0, 0, 292, 292, 292, 292]]).T
sys = OdometrySystem(alphas, beta)

# setup places to put things
all_z = np.zeros((times, steps, 2, 3))
all_u = np.zeros((times, steps, 3))
all_x = np.zeros((times, steps, 3))
start = {}
middle = {}
end = {}
for m in range(moments):
    num = len(list(cwr(np.arange(3), m+1)))
    start[m]  = np.zeros((times, steps, num))
    middle[m] = np.zeros((times, steps, num))
    end[m]    = np.zeros((times, steps, num))

bar = tqdm(total=times*steps)

bad_time = []
bad_step = []

for i in range(times):
    xs = data[0,i]['realRobot'][0,0].T
    us = data[0,i]['noisefreeControl'][0,0].T
    zs = data[0,i]['realObservation'][0,0].T

    all_z[i] = zs
    all_u[i] = us
    all_x[i] = xs

    n = 1000
    # pf = ParticleFilter(sys, N=n, mean=np.array([180, 50, 0]), cov=np.diag([200, 200, np.pi/4]), pz_x=sys.pz_x)
    pf = ParticleFilter(sys, N=n, mean=xs[0], cov=np.diag([0.1, 0.1, 0.1]), pz_x=sys.pz_x)
    all_particles = []

    for j, (u, z) in enumerate(zip(us, zs)):
        # save where we started at each time step
        # somewhat redundant, but makes things a bit easier later
        for m in range(moments):
            start[m][i,j] = calc_moment(pf.particles, m+1)

        # predict step and save
        p = pf.predict(u)
        for m in range(moments):
            middle[m][i,j] = calc_moment(p, m+1)

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

        for m in range(moments):
            end[m][i,j] = calc_moment(p, m+1)

        bar.update(1)
    
    # import matplotlib.pyplot as plt
    # plt.plot(xs[:,0], xs[:,1], label="Actual")
    # plt.plot(start_mu[steps*i:steps*(i+1),0], start_mu[steps*i:steps*(i+1),1], label="PF")
    # plt.legend()
    # plt.show()
    
with open(f'exact_data_{moment}.pkl', 'rb') as handle:
    pickle.dump([start, middle, end], handle, protocol=pickle.HIGHEST_PROTOCOL)