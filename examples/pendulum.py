import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy.linalg import inv, det

from systems import DiscreteNonlinearSystem
from filters import ExtendKalmanFilter, ParticleFilter, UnscentedKalmanFilter

from scipy.stats import norm
from time import time

############   setup simple pendulum as test     ##################
dt = .1
b = 1
c = 1
u_0, x_0, x_1 = sy.symbols('u_0 x_0 x_1')
f = sy.Array([x_0+dt*x_1, -dt*sy.sin(x_0) + (1-dt*b)*x_1 + c*u_0])
h = sy.Array([x_0])

R = np.array([[.01, 0],
                [0, .001]])
Q = np.array([[.1]])

############   make system and gather data     ##################
t = 500
x0 = np.array([2,0])

sys = DiscreteNonlinearSystem(f, h, R, Q, setup_parallel=False)
u = lambda t : np.array([np.cos(t/20)/20])
x, u, z = sys.gen_data(x0, t, u, noise=True)
x_perf, _, _ = sys.gen_data(x0, t, u, noise=False)

############     Run it through the ekf       ##################
sigma = np.eye(2)*.1
def run_ekf(plot=False):
    ekf = ExtendKalmanFilter(sys, x0, sigma)
    x_clean, sigmas = ekf.iterate(u, z)
    sig_x = np.sqrt(sigmas[:,0,0])

    if plot:
        plt.plot(t, x_clean.T[0], label="EKF Filtered Data")
        plt.fill_between(t, x_clean.T[0]-3*sig_x, x_clean.T[0]+3*sig_x, alpha=0.3)

    return x_clean

############     Run it through the ukf       ##################
def run_ukf(plot=False):
    ukf = UnscentedKalmanFilter(sys, x0, sigma)
    x_clean, sigmas = ukf.iterate(u, z)
    sig_x = np.sqrt(sigmas[:,0,0])

    if plot:
        plt.plot(t, x_clean.T[0], label="UKF Filtered Data")
        plt.fill_between(t, x_clean.T[0]-3*sig_x, x_clean.T[0]+3*sig_x, alpha=0.3)

    return x_clean

############     Run it through the pf        ##################
def run_pf(n=100, plot=False):
    pf = ParticleFilter(sys, N=100, sample='normal', mean=x0, cov=sigma)
    all_particles = []
    for us, zs in zip(u, z):
        pf.update(us)
        all_particles.append( pf.predict(zs) )
    all_particles = np.array(all_particles)

    if plot:
        plt.plot(t, all_particles.mean(axis=1).T[0], label="PF Filtered Data")
        for h, particles in zip(t, all_particles):
            plt.scatter(np.ones_like(particles.T[0])*h, particles.T[0], c='r', alpha=0.1, s=1)

    return all_particles
############            plot             ##################

t = np.arange(t)*dt
plt.plot(t, x_perf.T[0], label="Noiseless Data")
plt.plot(t, x.T[0], label="OG Data")
# plt.plot(t, z, label="Measurement Data", alpha=0.3)
run_ukf(plot=True)
plt.legend()
plt.show()

# all_particles = run_pf(10000)
# all_ekf = run_ekf()
# plt.plot(t, np.abs(x.T[0]-all_particles.mean(axis=1).T[0]), label='pf')
# plt.plot(t, np.abs(x.T[0]-all_ekf.T[0]), label='ekf')
# plt.legend()
# plt.show()