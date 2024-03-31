'''
Author: Chris Xiao yl.xiao@mail.utoronto.ca
Date: 2024-03-31 01:27:47
LastEditors: Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime: 2024-03-31 01:53:12
FilePath: /CPD-Pytorch/examples/bunny_rigid_3D.py
Description: 3D rigid registration example using the bunny dataset.
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
'''
import argparse
from functools import partial
import matplotlib.pyplot as plt
from torchcpd import RigidRegistration
import numpy as np
import os
import torch as th


def visualize(iteration, error, X, Y, ax, fig, save_fig=False):
    plt.cla()
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    th.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error[0]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    ax.view_init(90, -90)
    if save_fig is True:
        ax.set_axis_off()

    plt.draw()
    if save_fig is True:
        os.makedirs("../images/rigid_bunny/", exist_ok=True)
        fig.savefig("../images/rigid_bunny/rigid_bunny_3D_{:04}.tiff".format(iteration), dpi=600)  # Used for making gif.
    plt.pause(0.001)


def main():
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'
    X = np.loadtxt('../data/bunny_target.txt')
    # synthetic data, equaivalent to X + 1
    Y = np.loadtxt('../data/bunny_source.txt')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    callback = partial(visualize, ax=ax, fig=fig, save_fig=False)

    reg = RigidRegistration(**{'X': X, 'Y': Y, 'device': device})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    main()
