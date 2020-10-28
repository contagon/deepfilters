import numpy as np
from numba import njit, guvectorize
import sympy as sy

from systems import mvn

class DiscreteNonlinearSystem:
    def __init__(self, f, h, R, Q, var=None, setup_parallel=False):
        self.n = len(R)
        self.m = len(Q)
        self.R = R
        self.Q = Q

        # convert sympy functions into jitted functions
        # we use sympy so it can be used with EKF/Jacobians later
        if var is None:
            self.var = [f"x{i}" for i in range(self.n)]
        else:
            self.var = var
        self.f_sym = f
        self.h_sym = h
        self.f = sy.lambdify([self.var], f, 'numpy')
        self.h = sy.lambdify([self.var], h, 'numpy')

        # setup the parrallelized datagen function
        self.parallel = setup_parallel
        if setup_parallel:
            self.f = njit(self.f)
            self.h = njit(self.h)
            self.setup_parallel()

    def setup_parallel(self):
        f = self.f
        h = self.h
        n = self.n
        m = self.m
        L = np.linalg.cholesky(self.R)
        P = np.linalg.cholesky(self.Q)

        @guvectorize(["f8[:], i8[:], i8[:], boolean, f8[:,:], f8[:,:]"], 
                            "(n),(t),(m),() -> (t,n),(t,m)", 
                                nopython=True, target='parallel')
        def gen_data_parallel(x0, t, m_array, noise, x, z):
            x[0] = x0
            z[0] = np.array(h(x0))
            if noise:
                z[0] += mvn(mean=np.zeros(m), sqrtcov=P)
            for i in t:
                if i == 0:
                    continue
                if noise:
                    x[i] = np.array(f(x[i-1])) + mvn(mean=np.zeros(n), sqrtcov=L)
                    z[i] = np.array(h(x[i])) + mvn(mean=np.zeros(m), sqrtcov=P)
                else:
                    x[i] = np.array(f(x[i-1]))
                    z[i] = np.array(h(x[i]))


        self._gen_data_parallel = gen_data_parallel

    def _gen_data_single(self, x0, t, L, P, noise=True):
        x = np.zeros((t, self.n))
        z = np.zeros((t, self.m))
        x[0] = x0
        z[0] = np.array(self.h(x0))
        if noise:
            z[0] += mvn(mean=np.zeros(self.m), sqrtcov=P)
        for i in range(1,t):
            if noise:
                x[i] = np.array(self.f(x[i-1])) + mvn(mean=np.zeros(self.n), sqrtcov=L)
                z[i] = np.array(self.h(x[i])) + mvn(mean=np.zeros(self.m), sqrtcov=P)
            else:
                x[i] = np.array(self.f(x[i-1]))
                z[i] = np.array(self.h(x[i]))
        return x, z

        


    def gen_data(self, x0, t, noise=True):
        """Wrapper to call parallelized version a little more intuitively

        Args:
            x0: numpy array size (N, n), with N being # of simulations, n size of state space
            t: number of timesteps to iterate
            noise: bool whether or not measurements/state is impacted by mean-zero noise

        Returns:
            x: generated data, size (N,t,n)
            z: generated measurements, size (N,t,m)"""

        if self.parallel:
            return self._gen_data_parallel(x0, np.arange(t), np.arange(self.m), noise)
        else:
            L = np.linalg.cholesky(self.R)
            P = np.linalg.cholesky(self.Q)

            #determine if we need to loop over multiple starting conditions
            if len(x0.shape) > 1:
                x = np.zeros((x0.shape[0], t, self.n))
                z = np.zeros((x0.shape[0], t, self.m))
                for i, x1 in enumerate(x0):
                    x[i], z[i] = self._gen_data_single(x1, t, L, P, noise)

                return x, z

            else:
                return self._gen_data_single(x0, t, L, P, noise)


if __name__ == "__main__":
    #setup simple pendulum as test
    dt = .1
    b = 1
    c = 1
    x0, x1 = sy.symbols('x0 x1')
    f = sy.Array([x0+dt*x1, -dt*sy.sin(x0) + (1-dt*b)*x1])
    h = sy.Array([x0])

    R = np.ones((2,2)) *.1
    Q = np.array([[.1]])

    sys = DiscreteNonlinearSystem(f, h, R, Q)

    x0 = np.random.random(size=(200,2))
    x, z =  sys.gen_data(x0, 1000)
    x, z =  sys.gen_data(x0, 1000, noise=False)

    print(x.shape, z.shape)