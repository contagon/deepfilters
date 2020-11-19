import numpy as np
from numba import njit

@njit
def mvn(mean, sqrtcov):
    u = np.zeros_like(mean)
    for i in range(len(mean)):
        u[i] = np.random.standard_normal()

    return mean + sqrtcov@u


if __name__ == "__main__":
    #Try it and plot as test
    import seaborn as sns
    import matplotlib.pyplot as plt

    mean = np.array([5,11]).astype('float64')
    cov = np.array([[2,1],
                    [1,2]]).astype('float64')

    samples = []
    for i in range(10000):
        samples.append( mvn(mean, cov) )
    
    samples = np.array(samples).T

    sns.jointplot(x=samples[0],
              y=samples[1], 
              kind="kde", 
              space=0)
    plt.show()