from filters import BayesianFilter

import numpy as np
from numpy.linalg import inv
from numba import njit, guvectorize

class UnscentedKalmanFilter(BayesianFilter):

    def __init__(self, sys, mu, sigma, kappa=0, alpha=1, beta=2):
        self.sys = sys
        self.mus = [mu]
        self.sigmas = [sigma]

        #initialize parameters
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = alpha**2*(self.sys.n + kappa) - self.sys.n

        #setup all the weights
        self.wm = np.zeros(2*self.sys.n + 1)
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.sys.n + self.lambda_)
        self.wc[0] = self.wm[0] + 1 - self.alpha**2 + self.beta
        self.wm[1:] = 1 / (2*(self.sys.n + self.lambda_))
        self.wc[1:] = 1 / (2*(self.sys.n + self.lambda_))
        self.gamma = np.sqrt(self.sys.n + self.lambda_)

    @property
    def mu(self):
        return self.mus[-1]

    @property
    def sigma(self):
        return self.sigmas[-1]

    def update(self, u):
        #get mubar and sigmabar
        sqrtSigma = np.linalg.cholesky(self.sigma)
        X = np.vstack((self.mu, self.mu+self.gamma*sqrtSigma.T, self.mu-self.gamma*sqrtSigma.T))
        Xbar, _, Zbar = self.sys.gen_data(X, 1, u, noise_x=False, noise_z=False, noise_u=False)
        Xbar = Xbar.squeeze(1)
        Zbar = Zbar.squeeze(1)
        
        mu_bar = Xbar.T@self.wm
        diff = Xbar - mu_bar
        sigma_bar = np.sum([wc*np.outer(d,d) for wc, d in zip(self.wc, diff)], axis=0) + self.sys.cov_x

        #save for use later
        self.mus.append( mu_bar )
        # print(self.sigma)
        self.sigmas.append( sigma_bar )

        return mu_bar, sigma_bar

    def predict(self, z):
        sqrtSigma = np.linalg.cholesky(self.sigma)
        Xbar = np.vstack((self.mu, self.mu+self.gamma*sqrtSigma.T, self.mu-self.gamma*sqrtSigma.T))
        Zbar = np.zeros((2*self.sys.n+1, self.sys.m))
        for i, x in enumerate(Xbar):
            Zbar[i] = self.sys.h(x)
        z_bar = Zbar.T@self.wm

        diffZ = Zbar - z_bar
        diffX = Xbar - self.mu
        S = np.sum([wc*np.outer(d,d) for wc, d in zip(self.wc, diffZ)], axis=0) + self.sys.cov_z
        sigma_xz = np.sum([wc*np.outer(c,d) for wc, c, d in zip(self.wc, diffX, diffZ)], axis=0)

        K = sigma_xz@inv(S)
        self.mus[-1] = self.mu + K@(z - z_bar)
        self.sigmas[-1] = self.sigma - K@S@K.T

        return self.mu, self.sigma

    def iterate(self, us, zs):
        """given a sequence of observation, iterate through EKF"""
        for u, z in zip(us, zs):
            self.update(u)
            self.predict(z)

        return np.array(self.mus)[1:], np.array(self.sigmas)[1:]