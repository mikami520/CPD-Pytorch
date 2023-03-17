from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchcpd import DeformableRegistration, RigidRegistration, AffineRegistration
import numpy as np
import torch as th

def visualize(iteration, error, X, Y, ax):
    
    plt.cla()
    X = X.detach().cpu()
    Y = Y.detach().cpu()
    th.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error[0]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    true_affine = False
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    fish_target = np.loadtxt('./fish_target.txt')
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2)).astype(np.float64)
    X = th.from_numpy(X).float().cuda()
    if true_affine is True:
        theta = np.pi / 6.0
        R = th.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]).float().cuda()
        t = th.tensor([0.5, 1.0, 0.0]).float().cuda()

        # Create shear matrix
        shear_matrix = th.tensor([[1, 0, 0.5], [0, 1, 4], [0, 1, 1]]).float().cuda()

        R = th.mm(R, shear_matrix)

        Y = th.add(th.mm(X, R), t)
    else:
        fish_source = np.loadtxt('./fish_source.txt')
        Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
        Y1[:, :-1] = fish_source
        Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
        Y2[:, :-1] = fish_source
        Y = np.vstack((Y1, Y2))
        Y = th.from_numpy(Y).float().cuda()
    '''
    if rigid is True:
        theta = np.pi/6
        R = th.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=th.float64).float().cuda()
        t = th.tensor([0.5, 1.0, 0.0], dtype=th.float64).float().cuda()
        Y = th.mm(X, R) + t
    else:
        fish_source = np.loadtxt('./fish_source.txt')
        Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
        Y1[:, :-1] = fish_source
        Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
        Y2[:, :-1] = fish_source
        Y = np.vstack((Y1, Y2)).astype(np.float64)
        Y = th.from_numpy(Y).float().cuda()
    '''
    '''
    fish_source = np.loadtxt('./fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2)).astype(np.float64)
    Y = th.from_numpy(Y).float().cuda()
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = AffineRegistration(**{'X': X, 'Y': Y, 'device': device})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()