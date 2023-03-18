"""
This is a pytorch implementation of the coherent point drift [CPD](https://arxiv.org/abs/0905.2635/)
algorithm by Myronenko and Song. It provides three registration methods for point clouds: 
1. Scale and rigid registration
2. Affine registration
3. Gaussian regularized non-rigid registration
Licensed under an Apache License (c) Version 2.0, January 2004 Yuliang Xiao.
Distributed here: https://github.com/mikami520/CPD-Pytorch
"""

from .rigidReg import RigidRegistration
from .affineReg import AffineRegistration
from .deformReg import gaussian_kernel, DeformableRegistration
from .constDeformReg import ConstrainedDeformableRegistration