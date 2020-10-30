import numpy as np
from numba import njit, guvectorize
import sympy as sy

from systems import mvn

class DiscreteNonlinearSystem:
    def __init__(self, f, h, R, Q, x_var=None, u_var=None, setup_parallel=False):
        self.n = len(R)
        self.m = len(Q)
        self.R = R
        self.Q = Q
        self.sqrtR = np.linalg.cholesky(self.R)
        self.sqrtQ = np.linalg.cholesky(self.Q)

        # convert sympy functions into jitted functions
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
        if free_symbols.difference( set(self.x_var + self.u_var) ) != set():
            raise ValueError("Have an extra variable in f")
        if set(str(i) for i in h.free_symbols).difference( set(self.x_var) ) != set():
            raise ValueError("Have an extra variable in h")

        self.f_sym = f
        self.h_sym = h
        self.f = sy.lambdify([self.x_var, self.u_var], f, 'numpy')
        self.h = sy.lambdify([self.x_var], h, 'numpy')

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
        sqrtR = self.sqrtR
        sqrtQ = self.sqrtQ

        @guvectorize(["f8[:], i8[:], f8[:,:], i8[:], boolean, f8[:,:], f8[:,:]"], 
                            "(n),(t),(t,k),(m),() -> (t,n),(t,m)", 
                                nopython=True, target='parallel')
        def gen_data_parallel(x0, t, u, m_array, noise, x, z):
            #get first iteration in
            if noise:
                x[0] = np.array(f(x0, u[0])) + mvn(mean=np.zeros(n), sqrtcov=sqrtR)
                z[0] = np.array(h(x[0])) + mvn(mean=np.zeros(m), sqrtcov=sqrtQ)
            else:
                x[0] = np.array(f(x0, u[0]))
                z[0] = np.array(h(x[0]))

            for i in t:
                if i == 0:
                    continue
                if noise:
                    x[i] = np.array(f(x[i-1], u[i])) + mvn(mean=np.zeros(n), sqrtcov=sqrtR)
                    z[i] = np.array(h(x[i])) + mvn(mean=np.zeros(m), sqrtcov=sqrtQ)
                else:
                    x[i] = np.array(f(x[i-1], u[i]))
                    z[i] = np.array(h(x[i]))


        self._gen_data_parallel = gen_data_parallel

    def _gen_data_single(self, x0, t, u, noise=True):
        x = np.zeros((t, self.n))
        z = np.zeros((t, self.m))

        #get first iteration done
        if noise:
            x[0] = np.array(self.f(x0, u[0])) + mvn(mean=np.zeros(self.n), sqrtcov=self.sqrtR)
            z[0] = np.array(self.h(x[0])) + mvn(mean=np.zeros(self.m), sqrtcov=self.sqrtQ)
        else:
            x[0] = np.array(self.f(x0, u[0]))
            z[0] = np.array(self.h(x[0]))

        #run the rest of them
        for i in range(1,t):
            if noise:
                x[i] = np.array(self.f(x[i-1], u[i-1])) + mvn(mean=np.zeros(self.n), sqrtcov=self.sqrtR)
                z[i] = np.array(self.h(x[i])) + mvn(mean=np.zeros(self.m), sqrtcov=self.sqrtQ)
            else:
                x[i] = np.array(self.f(x[i-1], u[i-1]))
                z[i] = np.array(self.h(x[i]))
        return x, u, z

    def gen_data(self, x0, t, u=None, noise=True):
        """Wrapper to call parallelized version a little more intuitively

        Args:
            x0: numpy array size (N, n), with N being # of simulations, n size of state space.
                x0 is not returned in x. It's used as a seed to initialize things.
            t: number of timesteps to iterate
            noise: bool whether or not measurements/state is impacted by mean-zero noise

        Returns:
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

        if self.parallel:
            x, z = self._gen_data_parallel(x0, np.arange(t), u, np.arange(self.m), noise)
            return x, u, z
        else:
            #determine if we need to loop over multiple starting conditions
            if len(x0.shape) > 1:
                x = np.zeros((x0.shape[0], t, self.n))
                z = np.zeros((x0.shape[0], t, self.m))
                for i, x1 in enumerate(x0):
                    x[i], _, z[i] = self._gen_data_single(x1, t, u, noise)

                return x, u, z

            else:
                return self._gen_data_single(x0, t, u, noise)


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