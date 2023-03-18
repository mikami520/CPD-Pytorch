from functools import partial
import matplotlib.pyplot as plt
from torchcpd import RigidRegistration
import numpy as np
import torch as th

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    th.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error[0]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main(true_rigid=True):
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    X = np.loadtxt('../data/fish_target.txt')
    if true_rigid is True:
        theta = np.pi / 6.0
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([0.5, 1.0])
        Y = np.dot(X, R) + t
    else:
        Y = np.loadtxt('../data/fish_source.txt')

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = RigidRegistration(**{'X': X, 'Y': Y, 'device': device})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main(true_rigid=True)
