import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import os
from systems import DiscreteNonlinearSystem
from filters import ExtendKalmanFilter

############   setup simple pendulum as test     ##################
dt = .1
b = 1
c = 1
u_0, x_0, x_1 = sy.symbols('u_0 x_0 x_1')
f = sy.Array([x_0+dt*x_1, -dt*sy.sin(x_0) + (1-dt*b)*x_1 + c*u_0])
h = sy.Array([x_0])

cov_x = np.array([[.0001, 0],
                [0, .0001]])
cov_z = np.array([[.5]])

############   make system and gather data     ##################
t = 500
x0 = np.zeros((2))
u = np.random.random((100,500,1))

sys = DiscreteNonlinearSystem(f, h, cov_x, cov_z, setup_parallel=True)
x, u, z = sys.gen_data(x0, t, u=u, noise=False)

print(x.shape, z.shape, u.shape)
print(x[0].shape)