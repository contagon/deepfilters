import matplotlib.pyplot as plt

def plot_trajectory(x, label=None, ax=None):
    if ax is None:
        plt.plot(x[:,0], x[:,1], label=label)
    else:
        ax.plot(x[:,0], x[:,1], label=label)

def plot_landmarks(l, ax=None):
    if ax is None:
        plt.scatter(l[:,0], l[:,1], s=50, alpha=0.8)
    else:
        ax.scatter(l[:,0], l[:,1], s=50, alpha=0.8)
