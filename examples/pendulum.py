import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from systems import DiscreteNonlinearSystem
from filters import ExtendKalmanFilter

############   setup simple pendulum as test     ##################
dt = .1
b = 1
c = 1
u_0, x_0, x_1 = sy.symbols('u_0 x_0 x_1')
f = sy.Array([x_0+dt*x_1, -dt*sy.sin(x_0) + (1-dt*b)*x_1 + c*u_0])
h = sy.Array([x_0])

R = np.array([[.0001, 0],
                [0, .0001]])
Q = np.array([[.5]])

############   make system and gather data     ##################
t = 1000
x0 = np.array([2,0])

sys = DiscreteNonlinearSystem(f, h, R, Q, setup_parallel=False)
u = lambda t : np.array([np.cos(t/20)/20])
x, u, z = sys.gen_data(x0, t, u, noise=True)

############     Run it through the ekf       ##################
sigma = np.eye(2)
ekf = ExtendKalmanFilter(sys, x0, sigma)
x_clean, sigmas = ekf.iterate(u, z[1:])

############            plot             ##################
plt.plot(np.arange(t)*dt, x.T[0], label="OG Data")
plt.plot(np.arange(t)*dt, x_clean.T[0], label="EKF Filtered Data")
plt.plot(np.arange(t)*dt, z, label="Measurement Data", alpha=0.5)
plt.legend()
plt.show()