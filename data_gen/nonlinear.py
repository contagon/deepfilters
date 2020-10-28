import numpy as np
from numba import njit, guvectorize
import sympy as sy

from sample import mvn

class DiscreteNonlinearSystem:
    def __init__(self, f, h, R, Q):
        self.n = len(R)
        self.m = len(Q)

        # convert sympy functions into jitted functions
        # we use sympy so it can be used with EKF/Jacobians later
        var = [f"x{i}" for i in range(self.n)]
        self.f_sym = f
        self.f = njit( sy.lambdify([var], f, modules='math') )
        self.h_sym = h
        self.h = njit( sy.lambdify([var], h, modules='math') )
        self.R = R
        self.Q = Q

        # setup the parrallelized datagen function
        self.setup_data_gen()

    def setup_data_gen(self):
        f = self.f
        h = self.h
        n = self.n
        m = self.m
        L = np.linalg.cholesky(self.R)
        P = np.linalg.cholesky(self.Q)

        @guvectorize(["f8[:], i8[:], i8[:], f8[:,:], f8[:,:]"], 
                            "(n),(t),(m) -> (t,n),(t,m)", 
                                nopython=True, target='parallel')
        def gen_data_parallel(x0, t, m_array, x, z):
            x[0] = x0
            z[0] = np.array(h(x0)) + mvn(mean=np.zeros(m), sqrtcov=P)
            for i in t:
                if i == 0:
                    continue
                x[i] = np.array(f(x[i-1])) + mvn(mean=np.zeros(n), sqrtcov=L)
                z[i] = np.array(h(x[i])) + mvn(mean=np.zeros(m), sqrtcov=P)
        
        @guvectorize(["f8[:], i8[:], i8[:], f8[:,:], f8[:,:]"], 
                            "(n),(t),(m) -> (t,n),(t,m)", 
                                nopython=True, target='parallel')
        def gen_data_parallel_nonoise(x0, t, m_array, x, z):
            x[0] = x0
            z[0] = np.array(h(x0))
            for i in t:
                if i == 0:
                    continue
                x[i] = np.array(f(x[i-1])) 
                z[i] = np.array(h(x[i]))

        self._gen_data_parallel = gen_data_parallel
        self._gen_data_parallel_nonoise = gen_data_parallel_nonoise

    def gen_data(self, x0, t, noise=True):
        """Wrapper to call parallelized version a little more intuitively

        Args:
            x0: numpy array size (N, n), with N being # of simulations, n size of state space
            t: number of timesteps to iterate
            noise: bool whether or not measurements/state is impacted by mean-zero noise

        Returns:
            x: generated data, size (N,t,n)
            z: generated measurements, size (N,t,m)"""
        if noise:
            return self._gen_data_parallel(x0, np.arange(t), np.arange(self.m))
        else:
            return self._gen_data_parallel_nonoise(x0, np.arange(t), np.arange(self.m))


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