from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchcpd import DeformableRegistration, RigidRegistration, AffineRegistration, ConstrainedDeformableRegistration
import numpy as np
import torch as th
import math

fish_source = np.loadtxt('./fish_source.txt')
marker_size = 100
N_pts_include = 61
IDs = [1,10,20,30]
IDs_Y = IDs + [fish_source.shape[0] + i for i in IDs]
IDs_X = IDs + [N_pts_include + i for i in IDs]

def visualize(iteration, error, X, Y, ax):
    
    plt.cla()
    '''
    ids_X = np.arange(0, X.shape[0])
    ids_X = np.delete(ids_X, IDs_X)

    ids_Y = np.arange(0, Y.shape[0])
    ids_Y = np.delete(ids_Y, IDs_Y)
    '''
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    th.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    '''
    ax.scatter(X[ids_X, 0],  X[ids_X, 1], X[ids_X, 2], color='red', label='Target')
    ax.scatter(Y[ids_Y, 0],  Y[ids_Y, 1], Y[ids_Y, 2], color='blue', label='Source')

    ax.scatter(X[IDs_X, 0],  X[IDs_X, 1], X[IDs_X, 2], color='red', label='Target Constrained', s=marker_size, facecolors='none')
    ax.scatter(Y[IDs_Y, 0],  Y[IDs_Y, 1], Y[IDs_Y, 2], color='green', label='Source Constrained', s=marker_size, marker=(5, 1))
    '''
    #print(iteration, error[0])
    '''
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    '''
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error[0]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')

    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    true_affine = False
    rigid = False
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    fish_target = np.loadtxt('./fish_target.txt')
    #fish_target = fish_target[:N_pts_include]
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))
    '''
    if true_affine is True:
        theta = np.pi / 6.0
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        t = np.array([0.5, 1.0, 0.0])

        # Create shear matrix
        shear_matrix = [[1, 0, 0.5], [0, 1, 4], [0, 1, 1]]

        R = np.dot(R, shear_matrix)

        Y = np.dot(X, R) + t
    else:
        fish_source = np.loadtxt('./fish_source.txt')
        Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
        Y1[:, :-1] = fish_source
        Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
        Y2[:, :-1] = fish_source
        Y = np.vstack((Y1, Y2))

    '''
    if rigid is True:
        theta = np.pi / 6.0
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        t = np.array([0.5, 1.0, 0.0])
        Y = np.dot(X, R) + t
    else:
        fish_source = np.loadtxt('./fish_source.txt')
        Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
        Y1[:, :-1] = fish_source
        Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
        Y2[:, :-1] = fish_source
        Y = np.vstack((Y1, Y2))

    '''
    fish_source = np.loadtxt('./fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2))
    '''
    '''



    # select fixed correspondences
    src_id = th.tensor(IDs_Y, dtype=th.int64).to(device)
    tgt_id = th.tensor(IDs_X, dtype=th.int64).to(device)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = RigidRegistration(**{'X': X, 'Y': Y, 'device': device})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()