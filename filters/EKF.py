from filters import BayesianFilter

import numpy as np
from numpy.linalg import inv
from numba import njit, guvectorize
import sympy as sy

class ExtendKalmanFilter(BayesianFilter):

    def __init__(self, sys, mu, sigma):
        self.sys = sys
        self.mus = [mu]
        self.sigmas = [sigma]

        var = [f"x{i}" for i in range(self.sys.n)]
        self.F_sym = sy.Matrix(sys.f_sym.tolist()).jacobian(var)
        self.F = njit( sy.lambdify([var], self.F_sym, "numpy")  )
        self.H_sym = sy.Matrix(sys.h_sym.tolist()).jacobian(var)
        self.H = njit( sy.lambdify([var], self.H_sym, "numpy") )

    @property
    def mu(self):
        return self.mus[-1]

    @property
    def sigma(self):
        return self.sigmas[-1]

    def update(self, u):
        F = self.F(self.mu)
        #get mubar and sigmabar
        mu_bar = np.array(self.sys.f(self.mu))
        sigma_bar = F@self.sigma@F.T + self.sys.R

        #save for use later
        self.mus.append( mu_bar )
        self.sigmas.append( sigma_bar )

        return mu_bar, sigma_bar

    def predict(self, z):
        H = self.H(self.mu)
        zbar = self.sys.h(self.mu)
        #update
        K = self.sigma@H.T@inv( H@self.sigma@H.T + self.sys.Q )
        self.mus[-1] = self.mu + K@(z - zbar)
        self.sigmas[-1] = (np.eye(self.sys.n) - K@H)@self.sigma

        return self.mu, self.sigma

    def iterate(self, zs):
        """given a sequence of observation, iterate through EKF"""
        for z in zs:
            self.update(0)
            self.predict(z)

        return np.array(self.mus), np.array(self.sigmas)