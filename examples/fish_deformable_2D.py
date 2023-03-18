from functools import partial
import matplotlib.pyplot as plt
from torchcpd import DeformableRegistration
import numpy as np
import time
import torch as th


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    th.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    X = np.loadtxt('../data/fish_target.txt')
    Y = np.loadtxt('../data/fish_source.txt')

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = DeformableRegistration(**{'X': X, 'Y': Y, 'device': device})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
