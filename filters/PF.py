from filters import BayesianFilter

import numpy as np
from numpy.linalg import inv, det
from numba import njit, guvectorize
import sympy as sy

np.set_printoptions(precision=4, suppress=True)

class ParticleFilter(BayesianFilter):

    def __init__(self, sys, N=0, bounds=None, mean=None, cov=None, sample='normal', particles=None):
        self.sys = sys
        if not self.sys.parallel:
            self.sys.setup_parallel()
            self.sys.parallel = True

        #if particles aren't manually passed in, sample them here
        if particles is None:
            if sample == 'normal':
                self.particles = np.random.multivariate_normal(mean=mean, cov=cov, size=N)
            elif sample == 'uniform':
                self.particles = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(N, self.sys.n))
        else:
            self.particles = particles
        self.N = self.particles.shape[0]

        #set up p(z|x) so it's parallelize and super fast
        h = self.sys.h
        invQ = inv(self.sys.Q)
        norm = (2*np.pi)**(-self.sys.m/2) / np.sqrt(det(self.sys.Q))

        @guvectorize(["f8[:], f8[:], f8[:]"], "(m),(n)->()", nopython=True, target='parallel')
        def pz_x(z, x, out):
            diff = np.array(h(x)) - z
            out[0] = norm * np.exp( -diff.T@invQ@diff/2 )
        
        self.pz_x = pz_x

    def update(self, u):
        self.particles, _, _ = self.sys.gen_data(self.particles, 1, u)
        #squeeze to take out t dimension
        self.particles = self.particles.squeeze()

        return self.particles

    def predict(self, z):
        #update weights
        w = self.pz_x(z, self.particles).squeeze()
        w /= np.sum(w)

        #resample
        self.particles = self.particles[ np.random.choice(self.N, self.N, p=w) ]

        return self.particles

    def iterate(self, us, zs):
        """given a sequence of observation, iterate through PF"""
        for u, z in zip(us, zs):
            self.update(u)
            self.predict(z)