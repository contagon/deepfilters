import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from systems import DiscreteNonlinearSystem
from filters import ExtendKalmanFilter

#setup simple pendulum as test
dt = .1
b = 1
c = 1
x0, x1 = sy.symbols('x0 x1')
f = sy.Array([x0+dt*x1, -dt*sy.sin(x0) + (1-dt*b)*x1])
h = sy.Array([x0**2 + x1*5])

R = np.array([[.0001, 0],
                [0, .0001]])
Q = np.array([[.01]])

#make system and gather data
sys = DiscreteNonlinearSystem(f, h, R, Q)

t = 100
x0 = np.array([1,0])
x, z = sys.gen_data(x0, t)


#Run it through the ekf
sigma = np.eye(2)
ekf = ExtendKalmanFilter(sys, x0, sigma)
x_clean, sigmas = ekf.iterate(z[1:])
print(x_clean.shape)

plt.plot(np.arange(t)*dt, x.T[0], label="OG Data")
plt.plot(np.arange(t)*dt, x_clean.T[0], label="EKF Filtered Data")
# plt.plot(np.arange(t)*dt, z, label="Measurement Data", alpha=0.5)
plt.legend()
plt.show()