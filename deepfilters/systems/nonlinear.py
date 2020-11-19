import numpy as np
from numba import njit, guvectorize
import sympy as sy

from .sample import mvn

class DiscreteNonlinearSystem:
    def __init__(self, f, h, cov_x=None, cov_z=None, cov_u=None, x_var=None, u_var=None, setup_parallel=False):
        self.cov_x = cov_x
        self.cov_z = cov_z
        self.cov_u = cov_u
        if self.cov_x is not None:
            self.sqrtcov_x = np.linalg.cholesky(self.cov_x)
        if self.cov_z is not None:
            self.sqrtcov_z = np.linalg.cholesky(self.cov_z)
        if self.cov_u is not None:
            self.sqrtcov_u = np.linalg.cholesky(self.cov_u)

        # convert sympy functions into functions
        # we use sympy so it can be used with EKF/Jacobians later
        #pull out x states
        free_symbols = set(str(i) for i in f.free_symbols)
        if x_var is None:
            self.x_var = sorted([i for i in free_symbols if i[0:2] == "x_"])
        else: 
            self.x_var = x_var
        #pull out inputs
        if u_var is None:
            self.u_var = sorted([i for i in free_symbols if i[0:2] == "u_"])
        else: 
            self.u_var = u_var
        self.k = len(self.u_var)
        self.n = len(self.x_var)

        #double check equations are good
        if free_symbols.difference( set(self.x_var + self.u_var) ) != set():
            raise ValueError("Have an extra variable in f")
        if set(str(i) for i in h.free_symbols).difference( set(self.x_var) ) != set():
            raise ValueError("Have an extra variable in h")

        self.f_sym = f
        self.h_sym = h
        self.f = sy.lambdify([self.x_var, self.u_var], f, 'numpy')
        self.h = sy.lambdify([self.x_var], h, 'numpy')

        self.m = len(self.h(np.zeros(self.n)))

        # setup the parrallelized datagen function
        self.parallel = setup_parallel
        if setup_parallel:
            self.setup_parallel()

    def setup_parallel(self):
        #jit our functions
        self.f = njit(self.f)
        self.h = njit(self.h)

        #make variables local and available
        f = self.f
        h = self.h
        n = self.n
        m = self.m
        k = self.k
        sqrtcov_x = self.sqrtcov_x if hasattr(self, "sqrtcov_x") else np.zeros((self.n, self.n))
        sqrtcov_z = self.sqrtcov_z if hasattr(self, "sqrtcov_z") else np.zeros((self.m, self.m))
        sqrtcov_u = self.sqrtcov_u if hasattr(self, "sqrtcov_u") else np.zeros((self.k, self.k))

        @guvectorize(["f8[:], i8[:], f8[:,:], i8[:], boolean, boolean, boolean, f8[:,:], f8[:,:], f8[:,:]"], 
                            "(n),(t),(t,k),(m),(),(),() -> (t,n),(t,k),(t,m)", 
                                nopython=True, target='parallel')
        def gen_data_parallel(x0, t, u, m_array, noise_x, noise_z, noise_u, x, u_out, z):
            #get first iteration in
            if noise_u:
                u[0] += mvn(np.zeros(k), sqrtcov=sqrtcov_u)
            x[0] = np.array(f(x0, u[0]))
            if noise_x:
                x[0] += mvn(np.zeros(n), sqrtcov=sqrtcov_x)
            z[0] = np.array(h(x[0]))
            if noise_z:
                z[0] += mvn(np.zeros(m), sqrtcov=sqrtcov_z)

            for i in t:
                if i == 0:
                    continue
                if noise_u:
                    u[i] += mvn(np.zeros(k), sqrtcov=sqrtcov_u)
                x[i] = np.array(f(x[i-1], u[i]))
                if noise_x:
                    x[i] += mvn(np.zeros(n), sqrtcov=sqrtcov_x)
                z[i] = np.array(h(x[i]))
                if noise_z:
                    z[i] += mvn(np.zeros(m), sqrtcov=sqrtcov_z)
            u_out = u

        self._gen_data_parallel = gen_data_parallel

    def _gen_data_single(self, x0, t, u, noise_x, noise_z, noise_u):
        x = np.zeros((t, self.n))
        z = np.zeros((t, self.m))

        #get first iteration done
        if noise_u:
            u[0] += mvn(np.zeros(self.k), sqrtcov=self.sqrtcov_u)
        x[0] = np.array(self.f(x0, u[0]))
        if noise_x:
            x[0] += mvn(np.zeros(self.n), sqrtcov=self.sqrtcov_x)
        z[0] = np.array(self.h(x[0]))
        if noise_z:
            z[0] += mvn(np.zeros(self.m), sqrtcov=self.sqrtcov_z)

        for i in range(1,t):
            if noise_u:
                u[i] += mvn(np.zeros(self.k), sqrtcov=self.sqrtcov_u)
            x[i] = np.array(self.f(x[i-1], u[i]))
            if noise_x:
                x[i] += mvn(np.zeros(self.n), sqrtcov=self.sqrtcov_x)
            z[i] = np.array(self.h(x[i]))
            if noise_z:
                z[i] += mvn(np.zeros(self.m), sqrtcov=self.sqrtcov_z)
        return x, u, z

    def gen_data(self, x0, t, u=None, noise_x=True, noise_z=True, noise_u=False):
        """Wrapper to call parallelized version a little more intuitively

        Args:
            x0: numpy array size (N, n), with N being # of simulations, n size of state space.
                x0 is not returned in x. It's used as a seed to initialize things.
            t: number of timesteps to iterate
            noise_x: bool whether or not state is impacted by mean-zero noise. Default True
            noise_z: bool whether or not measurement is impacted by mean-zero noise. Default True
            noise_u: bool whether or not control is impacted by mean-zero noise. Default False

        cov_xeturns:
            x: generated data, size (N,t,n)
            z: generated measurements, size (N,t,m)"""

        #lots of options for passing in u
        #as a function of t
        if hasattr(u, '__call__'):
            u = np.array([u(t) for t in range(t)])
        #as 0, in which case it'll default to 0 everywhere
        elif u is None:
            u = np.zeros((t, self.k))
        #as a 1D numpy array, in which case it's just repeated over and over
        elif len(u.shape) == 1:
            u = np.tile(u, (t,1))
        #as 2D numpy array with size (t, k)
        #as 3D numpy array with size (b, t, k) where b is number of batches to run

        if not self.parallel:
            #determine if we need to loop over multiple starting conditions
            if len(x0.shape) > 1 or len(u.shape) > 2:
                self.parallel = True
                self.setup_parallel()
            else:
                return self._gen_data_single(x0, t, u, noise_x, noise_z, noise_u)
        
        if self.parallel:
            return self._gen_data_parallel(x0, np.arange(t), u, np.arange(self.m), noise_x, noise_z, noise_u)


if __name__ == "__main__":
    #setup simple pendulum as test
    dt = .1
    b = 1
    c = 1
    x_0, x_1 = sy.symbols('x_0 x_1')
    f = sy.Array([x_0+dt*x_1, -dt*sy.sin(x_0) + (1-dt*b)*x_1])
    h = sy.Array([x_0])

    cov_x = np.ones((2,2)) *.1
    cov_z = np.array([[.1]])

    sys = DiscreteNonlinearSystem(f, h, cov_x, cov_z, setup_parallel=True)

    x0 = np.random.random(size=(200,2))
    x, u, z =  sys.gen_data(np.array([2,0]), 10)