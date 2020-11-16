import sympy as sy
import numpy as np
from numpy.linalg import inv, det, cholesky

from numba import njit, guvectorize
from deepfilters.systems.sample import mvn

class OdometrySystem:

    def __init__(self, alphas, beta):
        x, y, theta, dt, dr1, dr2 = sy.symbols('x, y, theta, dt, dr1, dr2')
        lx, ly = sy.symbols('lx, ly')
        self.x_var = [x, y, theta]
        self.u_var = [dr1, dt, dr2]
        self.l_var = [lx, ly]

        self.Q = np.diag(beta)
        self.cov_z = self.Q
        self.sqrtQ = cholesky(self.Q)
        self.alphas = alphas

        self.f_sym =  sy.Array([x + dt*sy.cos(theta + dr1),
                                y + dt*sy.sin(theta + dr1),
                                theta + dr1 + dr2])
        self.h_sym = sy.Array([sy.sqrt((x - lx)**2 + (y - ly)**2),
                                sy.atan2(ly-y, lx-x)-theta])
        
        self.f = njit( sy.lambdify([self.x_var, self.u_var], self.f_sym, 'math') )
        self.h = njit( sy.lambdify([self.x_var, self.l_var], self.h_sym, 'math') )

        self.f_np = njit( sy.lambdify([self.x_var, self.u_var], self.f_sym, 'numpy') )
        self.h_np = lambda x, l: np.array([ np.sqrt((x[0] - l[0])**2+(x[1] - l[1])**2), np.arctan2(l[1]-x[1], l[0]-x[0])-x[2]])

        self.k = len(self.u_var)
        self.n = len(self.x_var)
        self.m = 2

        # make things local for guvectorize
        invQ = inv(self.Q)
        norm = (2*np.pi)**(-self.m/2) / np.sqrt(det(self.Q))
        h = self.h
        self.parallel = True

        @guvectorize(["f8[:,:], f8[:], f8[:,:], f8[:]"], "(r,m),(n),(r,l)->()", nopython=True, target='parallel')
        def pz_x(zs, x, ls, out):
            out[:] = 1
            for l,z in zip(ls, zs):
                zhat = np.array(h(x,l))
                diff = zhat - z
                while diff[1] < -np.pi:
                    diff[1] += 2*np.pi
                while diff[1] > np.pi:
                    diff[1] -= 2*np.pi
                out[:] *= norm * np.exp( -diff.T@invQ@diff/2 )

        self.pz_x = pz_x

    def cov_x(self, u):
        return np.diag([self.alphas[0]*u[0]**2 + self.alphas[1]*u[1]**2,
                        self.alphas[2]*u[1]**2 + self.alphas[3]*(u[0]**2 + u[1]**2),
                        self.alphas[0]*u[2]**2 + self.alphas[1]*u[1]**2])

    def gen_data(self, x0, t=1, u=None, noise_x=False, noise_z=True, noise_u=True):
        #only going to set this up to take one time step, which is what we need at the moment
        x = np.zeros((len(x0), self.n))
        if noise_u:
            R = np.diag([self.alphas[0]*u[0]**2 + self.alphas[1]*u[1]**2,
                        self.alphas[2]*u[1]**2 + self.alphas[3]*(u[0]**2 + u[1]**2),
                        self.alphas[0]*u[2]**2 + self.alphas[1]*u[1]**2])
            sqrtR = np.linalg.cholesky(R)

        for i, x0_i in enumerate(x0):
            if noise_u:
                u_new = mvn(u , sqrtR)
            x[i] = np.array( self.f(x0_i, u_new) )

        return x, u, u

if __name__ == "__main__":
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    from deepfilters.filters import ParticleFilter
    from deepfilters.filters import ExtendKalmanFilter

    # setup all data
    data = loadmat("../../simple_example/data.mat")['data'][0,31]
    xs = data['realRobot'][0,0].T
    us = data['noisefreeControl'][0,0].T
    zs = data['realObservation'][0,0].T
    l = np.array([[21, 168.3333, 315.6667, 463, 463, 315.6667, 168.3333, 21],
                    [0, 0, 0, 0, 292, 292, 292, 292]]).T

    # setup covariances
    alphas = np.array([0.05, 0.001, 0.05, 0.01])**2
    beta = np.array([10, np.deg2rad(10)])**2
    plt.scatter(l[:,0], l[:,1], s=20)
    plt.plot(xs[:,0], xs[:,1])

    #setup system
    sys = OdometrySystem(alphas, beta)

    n = 1000
    pf = ParticleFilter(sys, N=n, mean=np.array([180, 50, 0]), cov=np.diag([200, 200, np.pi/4]), pz_x=sys.pz_x)
    all_particles = []
    for u, z in zip(us, zs):
        pf.predict(u)

        temp = z[ ~np.isnan(z).any(axis=1) ]
        ls = l[ temp[:,2].astype('int')-1 ]
        meas = temp[:,:2]

        if meas.shape != (0,):
            all_particles.append( pf.update(meas, ls) )
            
    all_particles = np.array(all_particles)

    mean = all_particles.mean(axis=1)

    plt.plot(mean[:,0], mean[:,1], label="PF Filtered Data")
    for i in range(n):
        plt.scatter(all_particles[:,i,0], all_particles[:,i,1], alpha=0.3, s=1, c='r')


    # ekf = ExtendKalmanFilter(sys, np.array([180, 50, 0]), np.diag([200, 200, np.pi/4]))
    # for u,z in zip(us, zs):
    #     ekf.predict(u)

    #     temp = z[ ~np.isnan(z).any(axis=1) ]
    #     ls = l[ temp[:,2].astype('int')-1 ]
    #     meas = temp[:,:2]

    #     for zi, li in zip(meas, ls):
    #         ekf.update(zi, li)

    # plt.plot(np.array(ekf.mus)[:,0], np.array(ekf.mus)[:,1])

    
    plt.show()
